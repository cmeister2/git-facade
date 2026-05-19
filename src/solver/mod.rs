//! Brute-force solver for finding vanity SHA1 prefixes.

pub mod concurrent;
pub mod singlethreaded;
pub mod template;

use crate::digest::HexObjectDigest;
use crate::digest::ObjectDigest;

/// A solved commit object with its raw bytes, payload, and hex-encoded hash.
pub struct CommitObject {
    /// The full git object bytes (including `commit <len>\0` prefix).
    pub raw: Vec<u8>,
    /// The payload portion (without the prefix).
    pub payload: Vec<u8>,
    /// The hex-encoded SHA1 hash.
    pub hash: HexObjectDigest,
}

/// Trait for solvers that find a salt producing a desired digest prefix.
pub trait DigestPrefixSolver {
    /// Finds a salt value such that the template's SHA1 starts with `prefix`.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::ExhaustedSalts`] if no matching salt is found.
    fn solve(
        &self,
        template: &template::ObjectTemplate,
        prefix: &[u8],
    ) -> Result<CommitObject, SolverError>;
}

/// Errors that can occur during solving.
#[derive(Debug)]
pub enum SolverError {
    /// All possible salt values were tried without finding a match.
    ExhaustedSalts,
    /// An unexpected error occurred.
    Other(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::ExhaustedSalts => write!(f, "exhausted possible salts without finding a solution"),
            SolverError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SolverError {}

/// Checks whether a digest starts with the given byte prefix.
pub fn has_prefix(digest: &ObjectDigest, prefix: &[u8]) -> bool {
    let mut sum: u8 = 0;
    for (i, &p) in prefix.iter().enumerate() {
        sum |= digest.0[i] ^ p;
    }
    sum == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_prefix_match() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(has_prefix(&digest, &[0xc0, 0xff, 0xee]));
    }

    #[test]
    fn test_has_prefix_no_match() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xed, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(!has_prefix(&digest, &[0xc0, 0xff, 0xee]));
    }

    #[test]
    fn test_has_prefix_empty() {
        let digest = ObjectDigest([0xab; 20]);
        assert!(has_prefix(&digest, &[]));
    }

    #[test]
    fn test_has_prefix_single_byte() {
        let digest = ObjectDigest([
            0x88, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(has_prefix(&digest, &[0x88]));
        assert!(!has_prefix(&digest, &[0x89]));
    }

    #[test]
    fn test_has_prefix_full_digest() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xee, 0xba, 0xdc, 0x0d, 0xe5, 0x00, 0x11, 0x22,
            0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ]);
        assert!(has_prefix(&digest, &digest.0));
    }
}
