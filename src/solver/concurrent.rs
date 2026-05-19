//! Multi-threaded brute-force solver using rayon.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::digest::hex_encode_digest;
use crate::solver::template::ObjectTemplate;
use crate::solver::{has_prefix, CommitObject, DigestPrefixSolver, SolverError};

/// Number of salts each work unit processes before returning to the pool.
const CHUNK_SIZE: u64 = 4096;

/// A multi-threaded solver that distributes salt chunks across rayon workers.
pub struct ConcurrentSolver;

impl Default for ConcurrentSolver {
    fn default() -> Self {
        Self
    }
}

impl ConcurrentSolver {
    /// Creates a new concurrent solver.
    pub fn new() -> Self {
        Self
    }
}

impl DigestPrefixSolver for ConcurrentSolver {
    fn solve(&self, template: &ObjectTemplate, prefix: &[u8]) -> Result<CommitObject, SolverError> {
        let found = Arc::new(AtomicBool::new(false));

        let num_chunks = (u64::MAX / CHUNK_SIZE) + 1;

        let result = (0..num_chunks).into_par_iter().find_map_any(|chunk_idx| {
            if found.load(Ordering::Relaxed) {
                return None;
            }

            let start = chunk_idx.saturating_mul(CHUNK_SIZE);
            let end = start.saturating_add(CHUNK_SIZE);

            let mut tpl = template.clone();
            for salt in start..end {
                if found.load(Ordering::Relaxed) {
                    return None;
                }

                tpl.set_salt(salt);
                let digest = tpl.sum();

                if has_prefix(&digest, prefix) {
                    found.store(true, Ordering::Relaxed);
                    let hex_digest = hex_encode_digest(&digest);
                    return Some(CommitObject {
                        raw: tpl.bytes.clone(),
                        payload: tpl.payload().to_vec(),
                        hash: hex_digest,
                    });
                }
            }

            None
        });

        result.ok_or(SolverError::ExhaustedSalts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::parse_git_commit_object;
    use crate::solver::template::prepare_template;

    const RAW_HEADER_AND_BODY_OBJECT: &str = "tree e57181f20b062532907436169bb5823b6af2f099\n\
        author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        \n\
        Initial commit\n\
        36abde0100000000";

    #[test]
    fn test_concurrent_solve_short_prefix() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = ConcurrentSolver::new();
        let prefix = [0x88, 0x70];

        let result = solver.solve(&tpl, &prefix).unwrap();
        let hash_str = result.hash.as_str();
        assert!(
            hash_str.starts_with("8870"),
            "expected hash to start with 8870, got {}",
            hash_str
        );
    }

    #[test]
    fn test_concurrent_solve_result_hash_matches_payload() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = ConcurrentSolver::new();
        let prefix = [0x88, 0x70];

        let result = solver.solve(&tpl, &prefix).unwrap();

        use sha1::{Digest, Sha1};
        let actual_hash = Sha1::digest(&result.raw);
        let actual_hex = hex::encode(actual_hash);
        assert_eq!(actual_hex, result.hash.as_str());
    }

    #[test]
    fn test_concurrent_solve_three_byte_prefix() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = ConcurrentSolver::new();
        let prefix = [0xca, 0xfe, 0x00];

        let result = solver.solve(&tpl, &prefix).unwrap();
        let hash_str = result.hash.as_str();
        assert!(
            hash_str.starts_with("cafe00"),
            "expected hash to start with cafe00, got {}",
            hash_str
        );
    }
}
