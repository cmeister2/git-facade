//! Object template construction for brute-forcing.

use sha1::{Digest, Sha1};

use crate::commit::Object;
use crate::digest::ObjectDigest;
use crate::signing;

/// Hex lookup table.
const HEXTABLE: &[u8; 16] = b"0123456789abcdef";

/// The header name used for the brute-force salt.
const SALT_HEADER_NAME: &str = "facadesalt";

/// A git object template with a mutable salt region for brute-forcing.
#[derive(Clone)]
pub struct ObjectTemplate {
    /// The full object bytes including `commit <len>\0` prefix.
    pub bytes: Vec<u8>,
    /// Offset where the payload starts (after the prefix).
    pub payload_offset: usize,
    /// Offset where the 16-char hex salt value starts.
    pub salt_offset: usize,
}

impl ObjectTemplate {
    /// Writes a hex-encoded u64 salt value into the template at the salt offset.
    pub fn set_salt(&mut self, salt: u64) {
        hex_encode_uint64(&mut self.bytes[self.salt_offset..], salt);
    }

    /// Computes the SHA1 digest of the full object bytes.
    pub fn sum(&self) -> ObjectDigest {
        let hash = Sha1::digest(&self.bytes);
        let mut digest = [0u8; 20];
        digest.copy_from_slice(&hash);
        ObjectDigest(digest)
    }

    /// Returns the payload portion of the object (without the git object prefix).
    pub fn payload(&self) -> &[u8] {
        &self.bytes[self.payload_offset..]
    }

    /// Creates an [`IncrementalHasher`] that precomputes SHA1 state for all
    /// blocks before the salt, so each attempt only hashes the tail.
    pub fn incremental_hasher(&self) -> IncrementalHasher {
        IncrementalHasher::new(self)
    }
}

/// SHA1 block size in bytes.
const SHA1_BLOCK_SIZE: usize = 64;

/// Precomputed SHA1 state that avoids rehashing bytes before the salt block.
#[derive(Clone)]
pub struct IncrementalHasher {
    /// SHA1 state after processing all complete blocks before the salt.
    prefix_state: Sha1,
    /// Byte offset where the suffix (remaining blocks) starts.
    suffix_start: usize,
}

impl IncrementalHasher {
    /// Creates a new incremental hasher from a template.
    fn new(template: &ObjectTemplate) -> Self {
        let suffix_start = (template.salt_offset / SHA1_BLOCK_SIZE) * SHA1_BLOCK_SIZE;
        let mut hasher = Sha1::new();
        hasher.update(&template.bytes[..suffix_start]);
        Self {
            prefix_state: hasher,
            suffix_start,
        }
    }

    /// Computes the SHA1 digest by cloning the prefix state and hashing
    /// only the remaining bytes (the block containing the salt + everything after).
    pub fn sum(&self, template: &ObjectTemplate) -> ObjectDigest {
        let mut hasher = self.prefix_state.clone();
        hasher.update(&template.bytes[self.suffix_start..]);
        let hash = hasher.finalize();
        let mut digest = [0u8; 20];
        digest.copy_from_slice(&hash);
        ObjectDigest(digest)
    }
}

/// Writes a u64 as 16 hex characters into `dst`.
fn hex_encode_uint64(dst: &mut [u8], src: u64) {
    dst[15] = HEXTABLE[(src & 0x0f) as usize];
    dst[14] = HEXTABLE[((src >> 4) & 0x0f) as usize];
    dst[13] = HEXTABLE[((src >> 8) & 0x0f) as usize];
    dst[12] = HEXTABLE[((src >> 12) & 0x0f) as usize];
    dst[11] = HEXTABLE[((src >> 16) & 0x0f) as usize];
    dst[10] = HEXTABLE[((src >> 20) & 0x0f) as usize];
    dst[9] = HEXTABLE[((src >> 24) & 0x0f) as usize];
    dst[8] = HEXTABLE[((src >> 28) & 0x0f) as usize];
    dst[7] = HEXTABLE[((src >> 32) & 0x0f) as usize];
    dst[6] = HEXTABLE[((src >> 36) & 0x0f) as usize];
    dst[5] = HEXTABLE[((src >> 40) & 0x0f) as usize];
    dst[4] = HEXTABLE[((src >> 44) & 0x0f) as usize];
    dst[3] = HEXTABLE[((src >> 48) & 0x0f) as usize];
    dst[2] = HEXTABLE[((src >> 52) & 0x0f) as usize];
    dst[1] = HEXTABLE[((src >> 56) & 0x0f) as usize];
    dst[0] = HEXTABLE[((src >> 60) & 0x0f) as usize];
}

/// Builds an [`ObjectTemplate`] from a parsed commit, ready for brute-forcing.
///
/// If the commit has a `gpgsig` header, re-signs it and places the brute-force
/// salt inside the PGP armor `Comment:` field. Otherwise uses the unsigned
/// `facadesalt` header approach.
///
/// # Errors
///
/// Returns an error if the template cannot be constructed.
pub fn prepare_template(commit_object: &Object) -> Result<ObjectTemplate, String> {
    if signing::is_signed(commit_object) {
        prepare_signed_template(commit_object)
    } else {
        prepare_unsigned_template(commit_object)
    }
}

/// Unsigned path: strips `facadesalt`, adds a fresh one with zeroed salt.
fn prepare_unsigned_template(commit_object: &Object) -> Result<ObjectTemplate, String> {
    let mut payload_buf = Vec::new();

    let salt_header_prefix = format!("{} ", SALT_HEADER_NAME);
    let salt_value = hex::encode([0u8; 8]); // 16 hex chars of zeros

    for header in &commit_object.headers {
        if !header.value.starts_with(&salt_header_prefix) {
            payload_buf.extend_from_slice(header.value.as_bytes());
            payload_buf.push(b'\n');
        }
    }

    payload_buf.extend_from_slice(salt_header_prefix.as_bytes());
    let salt_offset_in_payload = payload_buf.len();
    payload_buf.extend_from_slice(salt_value.as_bytes());
    payload_buf.extend_from_slice(b"\n\n");

    payload_buf.extend_from_slice(&commit_object.message);

    let object_prefix = format!("commit {}\x00", payload_buf.len());
    let payload_offset = object_prefix.len();

    let mut object_buf = Vec::new();
    object_buf.extend_from_slice(object_prefix.as_bytes());
    object_buf.extend_from_slice(&payload_buf);

    Ok(ObjectTemplate {
        bytes: object_buf,
        salt_offset: payload_offset + salt_offset_in_payload,
        payload_offset,
    })
}

/// Signed path: re-signs the commit and places the salt in the PGP armor
/// `Comment:` field inside the `gpgsig` header.
fn prepare_signed_template(commit_object: &Object) -> Result<ObjectTemplate, String> {
    let content = signing::signable_content(commit_object);
    let signature = signing::gpg_sign(&content)?;
    let gpgsig_header = signing::build_gpgsig_with_salt(&signature)?;

    let gpgsig_salt_offset = signing::salt_offset_in_gpgsig(&gpgsig_header)
        .ok_or_else(|| "failed to locate salt in gpgsig header".to_string())?;

    let mut payload_buf = Vec::new();

    let salt_header_prefix = format!("{} ", SALT_HEADER_NAME);
    for header in &commit_object.headers {
        if header.value.starts_with("gpgsig ") || header.value.starts_with(&salt_header_prefix) {
            continue;
        }
        payload_buf.extend_from_slice(header.value.as_bytes());
        payload_buf.push(b'\n');
    }

    let salt_offset_in_payload = payload_buf.len() + gpgsig_salt_offset;
    payload_buf.extend_from_slice(gpgsig_header.as_bytes());
    payload_buf.push(b'\n');

    payload_buf.push(b'\n');
    payload_buf.extend_from_slice(&commit_object.message);

    let object_prefix = format!("commit {}\x00", payload_buf.len());
    let payload_offset = object_prefix.len();

    let mut object_buf = Vec::new();
    object_buf.extend_from_slice(object_prefix.as_bytes());
    object_buf.extend_from_slice(&payload_buf);

    Ok(ObjectTemplate {
        bytes: object_buf,
        salt_offset: payload_offset + salt_offset_in_payload,
        payload_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::parse_git_commit_object;

    const RAW_HEADER_AND_BODY_OBJECT: &str = "tree e57181f20b062532907436169bb5823b6af2f099\n\
        author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        \n\
        Initial commit\n\
        36abde0100000000";

    #[test]
    fn test_prepare_template_structure() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        // The template should start with "commit <len>\0"
        let prefix_end = tpl.bytes.iter().position(|&b| b == 0).unwrap() + 1;
        assert_eq!(prefix_end, tpl.payload_offset);

        let prefix_str = std::str::from_utf8(&tpl.bytes[..prefix_end - 1]).unwrap();
        assert!(prefix_str.starts_with("commit "));
    }

    #[test]
    fn test_prepare_template_contains_salt_header() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let payload = std::str::from_utf8(tpl.payload()).unwrap();
        assert!(payload.contains("facadesalt 0000000000000000"));
    }

    #[test]
    fn test_set_salt() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let mut tpl = prepare_template(&obj).unwrap();

        tpl.set_salt(0x0123456789abcdef);
        let payload = std::str::from_utf8(tpl.payload()).unwrap();
        assert!(payload.contains("facadesalt 0123456789abcdef"));
    }

    #[test]
    fn test_set_salt_zero() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let mut tpl = prepare_template(&obj).unwrap();

        tpl.set_salt(0);
        let payload = std::str::from_utf8(tpl.payload()).unwrap();
        assert!(payload.contains("facadesalt 0000000000000000"));
    }

    #[test]
    fn test_set_salt_max() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let mut tpl = prepare_template(&obj).unwrap();

        tpl.set_salt(u64::MAX);
        let payload = std::str::from_utf8(tpl.payload()).unwrap();
        assert!(payload.contains("facadesalt ffffffffffffffff"));
    }

    #[test]
    fn test_sum_deterministic() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let hash1 = tpl.sum();
        let hash2 = tpl.sum();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_sum_changes_with_salt() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let mut tpl = prepare_template(&obj).unwrap();

        tpl.set_salt(0);
        let hash1 = tpl.sum();
        tpl.set_salt(1);
        let hash2 = tpl.sum();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_clone_independence() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let mut tpl = prepare_template(&obj).unwrap();
        let mut tpl2 = tpl.clone();

        tpl.set_salt(42);
        tpl2.set_salt(99);

        assert_ne!(tpl.sum(), tpl2.sum());
    }

    #[test]
    fn test_prepare_template_strips_existing_salt() {
        let raw = "tree e57181f20b062532907436169bb5823b6af2f099\n\
            author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
            committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
            facadesalt deadbeefcafebabe\n\
            \n\
            Initial commit\n\
            36abde0100000000";

        let obj = parse_git_commit_object(raw.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        let payload = std::str::from_utf8(tpl.payload()).unwrap();
        // Should contain exactly one facadesalt header (the newly added one)
        assert_eq!(
            payload.matches("facadesalt").count(),
            1,
            "should have exactly one facadesalt header"
        );
    }

    #[test]
    fn test_payload_offset_correct() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        // Payload should not include the "commit <len>\0" prefix
        let payload = tpl.payload();
        assert!(payload.starts_with(b"tree "));
    }

    #[test]
    fn test_object_prefix_length_matches_payload() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        // Parse the length from the prefix
        let null_pos = tpl.bytes.iter().position(|&b| b == 0).unwrap();
        let prefix_str = std::str::from_utf8(&tpl.bytes[..null_pos]).unwrap();
        let len_str = prefix_str.strip_prefix("commit ").unwrap();
        let declared_len: usize = len_str.parse().unwrap();

        let actual_payload_len = tpl.bytes.len() - tpl.payload_offset;
        assert_eq!(declared_len, actual_payload_len);
    }

    #[test]
    fn test_hex_encode_uint64_values() {
        let mut buf = [0u8; 16];

        hex_encode_uint64(&mut buf, 0);
        assert_eq!(&buf, b"0000000000000000");

        hex_encode_uint64(&mut buf, 0xdeadbeef);
        assert_eq!(&buf, b"00000000deadbeef");

        hex_encode_uint64(&mut buf, u64::MAX);
        assert_eq!(&buf, b"ffffffffffffffff");

        hex_encode_uint64(&mut buf, 0x0123456789abcdef);
        assert_eq!(&buf, b"0123456789abcdef");
    }

    #[test]
    fn test_go_parity_template_bytes() {
        // Verify that our Rust template produces the same bytes as the Go version
        // for the standard test fixture.
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();

        // The payload should be:
        // tree <hash>\n
        // author ...\n
        // committer ...\n
        // facadesalt 0000000000000000\n
        // \n
        // Initial commit\n
        // 36abde0100000000
        let payload = std::str::from_utf8(tpl.payload()).unwrap();
        let expected_payload = "tree e57181f20b062532907436169bb5823b6af2f099\n\
            author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
            committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
            facadesalt 0000000000000000\n\
            \n\
            Initial commit\n\
            36abde0100000000";
        assert_eq!(payload, expected_payload);

        // The full object should be "commit <len>\0" + payload
        let expected_full = format!("commit {}\x00{}", expected_payload.len(), expected_payload);
        assert_eq!(tpl.bytes, expected_full.as_bytes());
    }

    #[test]
    fn test_incremental_matches_full_hash() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let mut tpl = prepare_template(&obj).unwrap();
        let hasher = tpl.incremental_hasher();

        for salt in 0..1000 {
            tpl.set_salt(salt);
            let full = tpl.sum();
            let incremental = hasher.sum(&tpl);
            assert_eq!(
                full, incremental,
                "mismatch at salt {}: full={} incremental={}",
                salt, full, incremental
            );
        }
    }

    #[test]
    fn test_incremental_hasher_suffix_start() {
        let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
        let tpl = prepare_template(&obj).unwrap();
        let hasher = tpl.incremental_hasher();

        // suffix_start should be aligned to 64-byte block boundary
        assert_eq!(hasher.suffix_start % SHA1_BLOCK_SIZE, 0);
        // suffix_start should be <= salt_offset
        assert!(hasher.suffix_start <= tpl.salt_offset);
        // suffix_start should be within one block of salt_offset
        assert!(tpl.salt_offset - hasher.suffix_start < SHA1_BLOCK_SIZE);
    }
}
