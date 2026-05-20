//! SHA1 parity tests: verify GPU SHA1 matches the `sha1` crate.

use sha1::{Digest, Sha1};
use wgpu_sha1::{GpuSha1, GpuTemplate};

fn cpu_sha1(data: &[u8]) -> [u8; 20] {
    let hash = Sha1::digest(data);
    let mut out = [0u8; 20];
    out.copy_from_slice(&hash);
    out
}

const HEXTABLE: &[u8; 16] = b"0123456789abcdef";

fn hex_encode_u64(salt: u64) -> [u8; 16] {
    let mut buf = [0u8; 16];
    for (i, byte) in buf.iter_mut().enumerate() {
        let nibble = (salt >> (60 - i * 4)) & 0xf;
        *byte = HEXTABLE[nibble as usize];
    }
    buf
}

fn make_template_with_salt(template: &[u8], salt_offset: usize, salt: u64) -> Vec<u8> {
    let mut data = template.to_vec();
    let hex = hex_encode_u64(salt);
    data[salt_offset..salt_offset + 16].copy_from_slice(&hex);
    data
}

fn make_test_template() -> (Vec<u8>, usize) {
    let payload = b"tree e57181f20b062532907436169bb5823b6af2f099\n\
        author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
        facadesalt 0000000000000000\n\
        \n\
        Initial commit\n\
        36abde0100000000";

    let prefix = format!("commit {}\x00", payload.len());
    let mut data = Vec::new();
    data.extend_from_slice(prefix.as_bytes());
    data.extend_from_slice(payload);

    let salt_offset = data
        .windows(16)
        .position(|w| w == b"0000000000000000")
        .expect("salt placeholder not found");

    (data, salt_offset)
}

fn make_offset_template(total_len: usize, salt_offset: usize) -> Vec<u8> {
    assert!(salt_offset + 16 <= total_len);
    let mut data: Vec<u8> = (0..total_len)
        .map(|i| b'a' + ((i * 17 + total_len) % 26) as u8)
        .collect();
    data[salt_offset..salt_offset + 16].copy_from_slice(b"0000000000000000");
    data
}

#[test]
fn test_sha1_parity_with_template() {
    let gpu = match GpuSha1::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU test (no adapter): {}", e);
            return;
        }
    };

    let (template_bytes, salt_offset) = make_test_template();
    let gpu_template = GpuTemplate::from_bytes(&template_bytes, salt_offset);

    let test_salts: &[u64] = &[0, 1, 42, 0xdeadbeef, 0x0123456789abcdef, u64::MAX];

    let gpu_digests = gpu
        .compute_digests(&gpu_template, test_salts)
        .expect("GPU compute_digests failed");

    for (i, &salt) in test_salts.iter().enumerate() {
        let full_bytes = make_template_with_salt(&template_bytes, salt_offset, salt);
        let cpu_digest = cpu_sha1(&full_bytes);
        let gpu_digest = gpu_digests[i];

        assert_eq!(
            cpu_digest,
            gpu_digest,
            "SHA1 mismatch at salt {:#x}:\n  CPU: {}\n  GPU: {}",
            salt,
            hex::encode(cpu_digest),
            hex::encode(gpu_digest),
        );
    }
}

#[test]
fn test_sha1_parity_varied_salt_offsets() {
    let gpu = match GpuSha1::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU test (no adapter): {}", e);
            return;
        }
    };

    let test_salts: &[u64] = &[0, 1, 0x0123456789abcdef, u64::MAX];
    let cases = [
        (40, 0),
        (70, 1),
        (100, 60),
        (120, 63),
        (120, 64),
        (150, 124),
        (190, 130),
        (276, 191),
    ];

    for (total_len, salt_offset) in cases {
        let template_bytes = make_offset_template(total_len, salt_offset);
        let gpu_template = GpuTemplate::from_bytes(&template_bytes, salt_offset);

        let gpu_digests = gpu
            .compute_digests(&gpu_template, test_salts)
            .expect("GPU compute_digests failed");

        for (i, &salt) in test_salts.iter().enumerate() {
            let full_bytes = make_template_with_salt(&template_bytes, salt_offset, salt);
            let cpu_digest = cpu_sha1(&full_bytes);
            let gpu_digest = gpu_digests[i];

            assert_eq!(
                cpu_digest,
                gpu_digest,
                "SHA1 mismatch at len {}, offset {}, salt {:#x}:\n  CPU: {}\n  GPU: {}",
                total_len,
                salt_offset,
                salt,
                hex::encode(cpu_digest),
                hex::encode(gpu_digest),
            );
        }
    }
}

#[test]
fn test_find_prefix_matches_cpu() {
    let gpu = match GpuSha1::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping GPU test (no adapter): {}", e);
            return;
        }
    };

    let (template_bytes, salt_offset) = make_test_template();
    let gpu_template = GpuTemplate::from_bytes(&template_bytes, salt_offset);

    let prefix = [0x88u8, 0x70];

    let result = gpu
        .find_prefix(&gpu_template, &prefix, 0, 4096 * 1024)
        .expect("GPU find_prefix failed");

    let result = result.expect("GPU should find a match within 4M salts");

    let full_bytes = make_template_with_salt(&template_bytes, salt_offset, result.salt);
    let cpu_digest = cpu_sha1(&full_bytes);

    assert_eq!(
        cpu_digest[0], 0x88,
        "first byte should be 0x88, got {:#04x}",
        cpu_digest[0]
    );
    assert_eq!(
        cpu_digest[1], 0x70,
        "second byte should be 0x70, got {:#04x}",
        cpu_digest[1]
    );
}
