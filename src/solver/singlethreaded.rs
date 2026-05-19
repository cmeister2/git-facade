//! Single-threaded brute-force solver.

use crate::digest::hex_encode_digest;
use crate::solver::template::ObjectTemplate;
use crate::solver::{has_prefix, CommitObject, DigestPrefixSolver, SolverError};

/// A single-threaded brute-force solver that iterates over a salt range.
pub struct SingleThreadedSolver {
    /// Start of the salt range (inclusive).
    pub salt_start: u64,
    /// End of the salt range (exclusive).
    pub salt_end: u64,
}

impl Default for SingleThreadedSolver {
    fn default() -> Self {
        Self {
            salt_start: 0,
            salt_end: u64::MAX,
        }
    }
}

impl SingleThreadedSolver {
    /// Creates a new solver that searches the full u64 salt space.
    pub fn new() -> Self {
        Self {
            salt_start: 0,
            salt_end: u64::MAX,
        }
    }

    /// Creates a new solver that searches a specific salt range.
    pub fn with_range(salt_start: u64, salt_end: u64) -> Self {
        Self {
            salt_start,
            salt_end,
        }
    }
}

impl DigestPrefixSolver for SingleThreadedSolver {
    fn solve(&self, template: &ObjectTemplate, prefix: &[u8]) -> Result<CommitObject, SolverError> {
        let mut tpl = template.clone();
        let hasher = tpl.incremental_hasher();
        for salt in self.salt_start..self.salt_end {
            tpl.set_salt(salt);
            let digest = hasher.sum(&tpl);

            if has_prefix(&digest, prefix) {
                let hex_digest = hex_encode_digest(&digest);
                return Ok(CommitObject {
                    raw: tpl.bytes.clone(),
                    payload: tpl.payload().to_vec(),
                    hash: hex_digest,
                });
            }
        }

        Err(SolverError::ExhaustedSalts)
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
    fn test_solve_short_prefix() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = SingleThreadedSolver::with_range(0, 4096 * 1024);
        let prefix = [0x88, 0x70];

        let result = solver.solve(&tpl, &prefix).unwrap();
        assert_eq!(&result.hash.0[..4], b"8870");
    }

    #[test]
    fn test_solve_single_byte_prefix() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = SingleThreadedSolver::with_range(0, 4096);
        let prefix = [0x00];

        let result = solver.solve(&tpl, &prefix).unwrap();
        assert_eq!(&result.hash.0[..2], b"00");
    }

    #[test]
    fn test_solve_exhausted() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        // Very small range with unlikely prefix
        let solver = SingleThreadedSolver::with_range(0, 1);
        let prefix = [0xff, 0xff, 0xff, 0xff];

        let result = solver.solve(&tpl, &prefix);
        assert!(matches!(result, Err(SolverError::ExhaustedSalts)));
    }

    #[test]
    fn test_solve_result_hash_matches_payload() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = SingleThreadedSolver::with_range(0, 4096 * 1024);
        let prefix = [0x88, 0x70];

        let result = solver.solve(&tpl, &prefix).unwrap();

        // Verify that hashing the raw bytes produces the reported hash
        use sha1::{Digest, Sha1};
        let actual_hash = Sha1::digest(&result.raw);
        let actual_hex = hex::encode(actual_hash);
        assert_eq!(actual_hex, result.hash.as_str());
    }

    #[test]
    fn test_solve_parity_with_go_native_test() {
        // The Go native test solves prefix [0x88, 0x70] with salt range 0..4096*1024
        // and asserts the hash starts with "8870". We do the same.
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let solver = SingleThreadedSolver::with_range(0, 4096 * 1024);
        let prefix = [0x88, 0x70];

        let result = solver.solve(&tpl, &prefix).unwrap();
        let hash_str = result.hash.as_str();
        assert!(
            hash_str.starts_with("8870"),
            "expected hash to start with 8870, got {}",
            hash_str
        );
    }
}
