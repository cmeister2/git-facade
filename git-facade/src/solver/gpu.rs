//! GPU-accelerated solver using wgpu compute shaders.

use sha1::{Digest, Sha1};

use crate::digest::{ObjectDigest, hex_encode_digest};
use crate::solver::template::ObjectTemplate;
use crate::solver::{CommitObject, DigestPrefixSolver, HexPrefix, SolverError};

use wgpu_sha1::{DEFAULT_BATCH_SIZE, GpuSha1, GpuTemplate};

/// A solver that offloads SHA1 brute-forcing to the GPU via wgpu.
pub struct GpuSolver {
    /// The GPU SHA1 engine.
    gpu: GpuSha1,
}

impl GpuSolver {
    /// Creates a new GPU solver, initializing the wgpu device and shader pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU adapter is found or device creation fails.
    pub fn new() -> Result<Self, String> {
        let gpu = GpuSha1::new().map_err(|e| e.to_string())?;
        Ok(Self { gpu })
    }
}

impl DigestPrefixSolver for GpuSolver {
    fn solve(
        &self,
        template: &ObjectTemplate,
        prefix: &HexPrefix,
    ) -> Result<CommitObject, SolverError> {
        let gpu_template = GpuTemplate::from_bytes(&template.bytes, template.salt_offset);

        let batch_size = DEFAULT_BATCH_SIZE;
        let search = self.gpu.create_prefix_search(
            &gpu_template,
            &prefix.bytes,
            prefix.half_byte,
            batch_size,
        );
        let mut salt_base: u64 = 0;

        loop {
            let result = search
                .find_prefix(salt_base)
                .map_err(|e| SolverError::Other(e.to_string()))?;

            if let Some(found) = result {
                let mut tpl = template.clone();
                tpl.set_salt(found.salt);

                let cpu_hash = Sha1::digest(&tpl.bytes);
                let mut digest = [0u8; 20];
                digest.copy_from_slice(&cpu_hash);
                let obj_digest = ObjectDigest(digest);

                let full = if prefix.half_byte {
                    prefix.bytes.len() - 1
                } else {
                    prefix.bytes.len()
                };
                let matches = digest[..full] == prefix.bytes[..full]
                    && (!prefix.half_byte || (digest[full] & 0xF0) == (prefix.bytes[full] & 0xF0));

                if matches {
                    let hex_digest = hex_encode_digest(&obj_digest);
                    return Ok(CommitObject {
                        salt: found.salt,
                        raw: tpl.bytes.clone(),
                        payload: tpl.payload().to_vec(),
                        hash: hex_digest,
                    });
                }

                eprintln!(
                    "GPU reported match at salt {} but CPU verification failed, continuing",
                    found.salt
                );
            }

            salt_base = salt_base
                .checked_add(batch_size as u64)
                .ok_or(SolverError::ExhaustedSalts)?;
        }
    }
}
