//! SHA1 digest types and hex encoding/decoding.

use std::fmt;

/// Hex lookup table.
const HEXTABLE: &[u8; 16] = b"0123456789abcdef";

/// A raw 20-byte SHA1 object digest.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ObjectDigest(pub [u8; 20]);

/// A 40-byte hex-encoded SHA1 object digest.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HexObjectDigest(pub [u8; 40]);

/// Hex-encodes a raw digest into a hex digest.
pub fn hex_encode_digest(src: &ObjectDigest) -> HexObjectDigest {
    let mut dst = [0u8; 40];
    let mut j = 0;
    for &byte in &src.0 {
        dst[j] = HEXTABLE[(byte >> 4) as usize];
        dst[j + 1] = HEXTABLE[(byte & 0x0f) as usize];
        j += 2;
    }
    HexObjectDigest(dst)
}

/// Decodes a hex digest back into a raw digest.
///
/// # Errors
///
/// Returns an error if the hex string contains invalid characters.
pub fn hex_decode_digest(src: &HexObjectDigest) -> Result<ObjectDigest, String> {
    let mut dst = [0u8; 20];
    hex::decode_to_slice(src.0, &mut dst)
        .map_err(|_| format!("cannot decode hex string {:?}", std::str::from_utf8(&src.0)))?;
    Ok(ObjectDigest(dst))
}

impl fmt::Display for ObjectDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl fmt::Debug for ObjectDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectDigest({})", self)
    }
}

impl fmt::Display for HexObjectDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for HexObjectDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HexObjectDigest({})", self)
    }
}

impl HexObjectDigest {
    /// Returns this hex digest as a `&str`.
    ///
    /// # Panics
    ///
    /// Panics if the internal bytes are not valid UTF-8 (should never happen
    /// for hex-encoded digests).
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("hex digest is always valid utf-8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_encode_decode_roundtrip() {
        let original = ObjectDigest([
            0xc0, 0xff, 0xee, 0xba, 0xdc, 0x0d, 0xe5, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        ]);
        let hex = hex_encode_digest(&original);
        assert_eq!(hex.as_str(), "c0ffeebadc0de500112233445566778899aabbcc");

        let decoded = hex_decode_digest(&hex).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_hex_encode_all_zeros() {
        let digest = ObjectDigest([0u8; 20]);
        let hex = hex_encode_digest(&digest);
        assert_eq!(hex.as_str(), "0000000000000000000000000000000000000000");
    }

    #[test]
    fn test_hex_encode_all_ff() {
        let digest = ObjectDigest([0xff; 20]);
        let hex = hex_encode_digest(&digest);
        assert_eq!(hex.as_str(), "ffffffffffffffffffffffffffffffffffffffff");
    }

    #[test]
    fn test_display_object_digest() {
        let digest = ObjectDigest([
            0xc0, 0xff, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        assert_eq!(
            format!("{}", digest),
            "c0ffee0000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_hex_decode_invalid() {
        let bad = HexObjectDigest(*b"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz");
        assert!(hex_decode_digest(&bad).is_err());
    }
}
