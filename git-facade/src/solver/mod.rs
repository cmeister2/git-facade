//! Brute-force solver for finding vanity SHA1 prefixes.

pub mod concurrent;
pub mod gpu;
pub mod singlethreaded;
pub mod template;

use crate::digest::HexObjectDigest;
use crate::digest::ObjectDigest;

/// A parsed hex prefix that may end on a half-byte (nibble) boundary.
///
/// For even-length hex strings like "facade" (6 chars), `bytes` = `[0xfa, 0xca, 0xde]`
/// and `half_byte` = false. For odd-length strings like "facade0" (7 chars),
/// `bytes` = `[0xfa, 0xca, 0xde, 0x00]` with the last nibble shifted high,
/// and `half_byte` = true.
pub struct HexPrefix {
    /// Decoded prefix bytes. If `half_byte` is true, the last byte's low nibble
    /// is zero-padded and should be ignored during matching.
    pub bytes: Vec<u8>,
    /// Whether the prefix ends on a nibble boundary (odd number of hex chars).
    pub half_byte: bool,
}

impl HexPrefix {
    /// Creates a full-byte prefix (no half-byte nibble).
    pub fn full(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.to_vec(),
            half_byte: false,
        }
    }

    /// Creates a half-byte prefix (last byte's low nibble is ignored).
    pub fn half(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.to_vec(),
            half_byte: true,
        }
    }
}

/// A solved commit object with its raw bytes, payload, and hex-encoded hash.
pub struct CommitObject {
    /// The 64-bit salt encoded into the commit template.
    pub salt: u64,
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
        prefix: &HexPrefix,
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
            SolverError::ExhaustedSalts => {
                write!(f, "exhausted possible salts without finding a solution")
            }
            SolverError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SolverError {}

/// Checks whether a digest starts with the given hex prefix.
pub fn has_prefix(digest: &ObjectDigest, prefix: &HexPrefix) -> bool {
    let full_bytes = if prefix.half_byte {
        prefix.bytes.len() - 1
    } else {
        prefix.bytes.len()
    };

    let mut sum: u8 = 0;
    for (i, &p) in prefix.bytes[..full_bytes].iter().enumerate() {
        sum |= digest.0[i] ^ p;
    }
    if sum != 0 {
        return false;
    }

    if prefix.half_byte {
        return (digest.0[full_bytes] & 0xF0) == (prefix.bytes[full_bytes] & 0xF0);
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_prefix_match() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(has_prefix(&digest, &HexPrefix::full(&[0xc0, 0xff, 0xee])));
    }

    #[test]
    fn test_has_prefix_no_match() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xed, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(!has_prefix(&digest, &HexPrefix::full(&[0xc0, 0xff, 0xee])));
    }

    #[test]
    fn test_has_prefix_empty() {
        let digest = ObjectDigest([0xab; 20]);
        assert!(has_prefix(&digest, &HexPrefix::full(&[])));
    }

    #[test]
    fn test_has_prefix_single_byte() {
        let digest = ObjectDigest([
            0x88, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(has_prefix(&digest, &HexPrefix::full(&[0x88])));
        assert!(!has_prefix(&digest, &HexPrefix::full(&[0x89])));
    }

    #[test]
    fn test_has_prefix_full_digest() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xee, 0xba, 0xdc, 0x0d, 0xe5, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ]);
        assert!(has_prefix(&digest, &HexPrefix::full(&digest.0)));
    }

    #[test]
    fn test_has_prefix_half_byte_match() {
        let digest = ObjectDigest([
            0xfa, 0xca, 0xde, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(has_prefix(
            &digest,
            &HexPrefix::half(&[0xfa, 0xca, 0xde, 0x00])
        ));
    }

    #[test]
    fn test_has_prefix_half_byte_no_match() {
        let digest = ObjectDigest([
            0xfa, 0xca, 0xde, 0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(!has_prefix(
            &digest,
            &HexPrefix::half(&[0xfa, 0xca, 0xde, 0x00])
        ));
    }

    #[test]
    fn test_has_prefix_half_byte_ignores_low_nibble() {
        let digest = ObjectDigest([
            0xfa, 0xca, 0xde, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert!(has_prefix(
            &digest,
            &HexPrefix::half(&[0xfa, 0xca, 0xde, 0x00])
        ));
    }
}
