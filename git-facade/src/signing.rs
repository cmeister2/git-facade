//! GPG signing helpers for git commit objects.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::commit::Object;

/// The header name used for the unsigned brute-force salt.
const SALT_HEADER_NAME: &str = "facadesalt";
/// Git header prefix for GPG signatures.
const GPGSIG_HEADER_PREFIX: &str = "gpgsig ";
/// PGP ASCII armor begin marker.
const PGP_ARMOR_BEGIN: &str = "-----BEGIN PGP SIGNATURE-----";
/// PGP armor Comment header prefix.
const COMMENT_PREFIX: &str = "Comment: ";
/// Zeroed 16-char hex salt placeholder.
const SALT_PLACEHOLDER: &str = "0000000000000000";

/// Returns true if the parsed commit contains a `gpgsig` header.
pub fn is_signed(commit: &Object) -> bool {
    commit
        .headers
        .iter()
        .any(|h| h.value.starts_with(GPGSIG_HEADER_PREFIX))
}

/// Builds the commit payload that GPG signs — all headers except `gpgsig` and
/// `facadesalt`, followed by a blank line and the message.
pub fn signable_content(commit: &Object) -> Vec<u8> {
    let mut buf = Vec::new();
    for header in &commit.headers {
        if header.value.starts_with(GPGSIG_HEADER_PREFIX)
            || header.value.starts_with(&format!("{} ", SALT_HEADER_NAME))
        {
            continue;
        }
        buf.extend_from_slice(header.value.as_bytes());
        buf.push(b'\n');
    }
    buf.push(b'\n');
    buf.extend_from_slice(&commit.message);
    buf
}

/// Signs content with GPG using the key from `git config user.signingkey`.
///
/// # Errors
///
/// Returns an error if the signing key is not configured, the GPG program
/// cannot be spawned, or GPG exits with a non-zero status.
pub fn gpg_sign(content: &[u8]) -> Result<String, String> {
    let key = get_signing_key()?;
    let gpg_program = get_gpg_program();

    let mut child = Command::new(&gpg_program)
        .args(["-bsau", &key])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {}", gpg_program, e))?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| "gpg stdin not available".to_string())?
        .write_all(content)
        .map_err(|e| format!("failed to write to gpg stdin: {}", e))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for gpg: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("gpg signing failed: {}", stderr));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("gpg produced invalid UTF-8: {}", e))
}

/// Builds a `gpgsig` header value with a `Comment:` salt line inserted into
/// the PGP armor block. Returns the full multi-line header string including
/// the `gpgsig ` prefix, with continuation lines prefixed by a space.
///
/// # Errors
///
/// Returns an error if the signature does not contain a PGP armor begin line.
pub fn build_gpgsig_with_salt(signature: &str) -> Result<String, String> {
    let mut result = String::new();
    let mut first = true;
    let mut inserted_comment = false;

    for line in signature.lines() {
        if first {
            result.push_str(&format!("{}{}", GPGSIG_HEADER_PREFIX, line));
            first = false;
        } else {
            result.push('\n');
            result.push_str(&format!(" {}", line));
        }

        if !inserted_comment && line == PGP_ARMOR_BEGIN {
            result.push('\n');
            result.push_str(&format!(" {}{}", COMMENT_PREFIX, SALT_PLACEHOLDER));
            inserted_comment = true;
        }
    }

    if !inserted_comment {
        return Err("PGP signature missing BEGIN armor line".to_string());
    }

    Ok(result)
}

/// Reuses an existing `gpgsig` header value and inserts or replaces a
/// `Comment:` salt line inside the ASCII armor block.
///
/// # Errors
///
/// Returns an error if the header does not contain a PGP armor begin line.
pub fn reuse_gpgsig_with_salt(gpgsig_header: &str) -> Result<String, String> {
    let mut lines: Vec<String> = gpgsig_header.lines().map(|line| line.to_string()).collect();
    let mut begin_index = None;
    let mut comment_index = None;

    for (index, line) in lines.iter().enumerate() {
        let logical = if index == 0 {
            line.strip_prefix(GPGSIG_HEADER_PREFIX)
                .unwrap_or(line.as_str())
        } else {
            line.strip_prefix(' ').unwrap_or(line.as_str())
        };

        if logical == PGP_ARMOR_BEGIN {
            begin_index = Some(index);
        }
        if logical.starts_with(COMMENT_PREFIX) {
            comment_index = Some(index);
        }
    }

    let begin_index =
        begin_index.ok_or_else(|| "PGP signature missing BEGIN armor line".to_string())?;
    let replacement = format!(" {}{}", COMMENT_PREFIX, SALT_PLACEHOLDER);

    if let Some(comment_index) = comment_index {
        lines[comment_index] = replacement;
    } else {
        lines.insert(begin_index + 1, replacement);
    }

    Ok(lines.join("\n"))
}

/// Returns the byte offset of the salt placeholder within the gpgsig header
/// string produced by [`build_gpgsig_with_salt`].
pub fn salt_offset_in_gpgsig(gpgsig_header: &str) -> Option<usize> {
    let comment_tag = format!(" {}{}", COMMENT_PREFIX, SALT_PLACEHOLDER);
    gpgsig_header
        .find(&comment_tag)
        .map(|pos| pos + 1 + COMMENT_PREFIX.len()) // skip the leading space and "Comment: "
}

/// Reads the GPG signing key from `git config user.signingkey`.
fn get_signing_key() -> Result<String, String> {
    let output = Command::new("git")
        .args(["config", "user.signingkey"])
        .output()
        .map_err(|e| format!("failed to read git config user.signingkey: {}", e))?;

    if !output.status.success() {
        return Err("git config user.signingkey is not set".to_string());
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Reads the GPG program from `git config gpg.program`, defaulting to `gpg`.
fn get_gpg_program() -> String {
    Command::new("git")
        .args(["config", "gpg.program"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "gpg".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::Header;

    #[test]
    fn test_build_gpgsig_with_salt() {
        let sig = "-----BEGIN PGP SIGNATURE-----\n\
                    \n\
                    iQIzBAAB\n\
                    =XXXX\n\
                    -----END PGP SIGNATURE-----";
        let result = build_gpgsig_with_salt(sig).unwrap();
        assert!(result.starts_with("gpgsig -----BEGIN PGP SIGNATURE-----\n"));
        assert!(result.contains(&format!(" Comment: {}", SALT_PLACEHOLDER)));
        // Every line after the first should start with a space
        for (i, line) in result.lines().enumerate() {
            if i > 0 {
                assert!(
                    line.starts_with(' '),
                    "continuation line {} does not start with space: {:?}",
                    i,
                    line
                );
            }
        }
    }

    #[test]
    fn test_salt_offset_in_gpgsig() {
        let sig = "-----BEGIN PGP SIGNATURE-----\n\niQIzBAAB\n=XXXX\n-----END PGP SIGNATURE-----";
        let gpgsig = build_gpgsig_with_salt(sig).unwrap();
        let offset = salt_offset_in_gpgsig(&gpgsig).unwrap();
        assert_eq!(
            &gpgsig[offset..offset + 16],
            SALT_PLACEHOLDER,
            "offset should point at the 16-char salt"
        );
    }

    #[test]
    fn test_build_gpgsig_missing_armor() {
        let sig = "not a PGP signature";
        assert!(build_gpgsig_with_salt(sig).is_err());
    }

    #[test]
    fn test_reuse_gpgsig_with_salt_inserts_comment() {
        let original = "gpgsig -----BEGIN PGP SIGNATURE-----\n \
\n \
 iQIzBAAB\n \
 =XXXX\n \
 -----END PGP SIGNATURE-----";

        let updated = reuse_gpgsig_with_salt(original).unwrap();
        assert!(updated.contains(&format!(" {}{}", COMMENT_PREFIX, SALT_PLACEHOLDER)));

        let begin_pos = updated.find(PGP_ARMOR_BEGIN).unwrap();
        let comment_pos = updated
            .find(&format!(" {}{}", COMMENT_PREFIX, SALT_PLACEHOLDER))
            .unwrap();
        assert!(comment_pos > begin_pos);
    }

    #[test]
    fn test_reuse_gpgsig_with_salt_replaces_existing_comment() {
        let original = format!(
            "gpgsig -----BEGIN PGP SIGNATURE-----\n {}old-comment\n iQIzBAAB\n =XXXX\n -----END PGP SIGNATURE-----",
            COMMENT_PREFIX
        );

        let updated = reuse_gpgsig_with_salt(&original).unwrap();
        assert!(updated.contains(&format!(" {}{}", COMMENT_PREFIX, SALT_PLACEHOLDER)));
        assert!(!updated.contains("old-comment"));
    }

    #[test]
    fn test_signable_content_strips_gpgsig_and_facadesalt() {
        let obj = Object {
            object_type: "commit".to_string(),
            headers: vec![
                Header {
                    value: "tree abc123".to_string(),
                },
                Header {
                    value: "author Test <test@test.com> 1000000000 +0000".to_string(),
                },
                Header {
                    value: "committer Test <test@test.com> 1000000000 +0000".to_string(),
                },
                Header {
                    value: "gpgsig -----BEGIN PGP SIGNATURE-----\n sig data\n -----END PGP SIGNATURE-----".to_string(),
                },
                Header {
                    value: "facadesalt deadbeef12345678".to_string(),
                },
            ],
            message: b"test message\n".to_vec(),
        };

        let content = signable_content(&obj);
        let content_str = String::from_utf8(content).unwrap();
        assert!(content_str.contains("tree abc123\n"));
        assert!(content_str.contains("author Test"));
        assert!(content_str.contains("committer Test"));
        assert!(!content_str.contains("gpgsig"));
        assert!(!content_str.contains("facadesalt"));
        assert!(content_str.contains("\n\ntest message\n"));
    }
}
