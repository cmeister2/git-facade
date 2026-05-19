//! Git command wrappers.

use std::process::Command;

use crate::digest::HexObjectDigest;

/// Returns the hex digest of the current HEAD commit.
///
/// # Errors
///
/// Returns an error if `git rev-parse HEAD` fails or produces unexpected output.
pub fn get_head_digest() -> Result<HexObjectDigest, String> {
    let out = run_command("git", &["rev-parse", "HEAD"])?;
    let trimmed = out.trim();
    if trimmed.len() != 40 {
        return Err(format!(
            "digest length not matching 40 != {}",
            trimmed.len()
        ));
    }
    let mut hex_digest = [0u8; 40];
    hex_digest.copy_from_slice(trimmed.as_bytes());
    Ok(HexObjectDigest(hex_digest))
}

/// Returns the raw contents of a commit object.
///
/// # Errors
///
/// Returns an error if `git cat-file -p` fails.
pub fn get_commit_contents(digest: &HexObjectDigest) -> Result<Vec<u8>, String> {
    let digest_str = digest.as_str();
    let out = run_command_raw("git", &["cat-file", "-p", digest_str])?;
    Ok(out)
}

/// Writes an object to the git store and returns its digest.
///
/// # Errors
///
/// Returns an error if `git hash-object` fails.
pub fn write_object(object_type: &str, contents: &[u8]) -> Result<HexObjectDigest, String> {
    let out = run_command_with_stdin(
        "git",
        &["hash-object", "-w", "-t", object_type, "--stdin"],
        contents,
    )?;
    let trimmed = out.trim();
    if trimmed.len() != 40 {
        return Err(format!(
            "unexpected hash-object output length: {}",
            trimmed.len()
        ));
    }
    let mut hex_digest = [0u8; 40];
    hex_digest.copy_from_slice(trimmed.as_bytes());
    Ok(HexObjectDigest(hex_digest))
}

/// Updates a git reference to point to a new hash.
///
/// # Errors
///
/// Returns an error if `git update-ref` fails.
pub fn update_reference(reference: &str, hash: &str) -> Result<(), String> {
    run_command("git", &["update-ref", reference, hash])?;
    Ok(())
}

/// Runs a command and returns stdout as a string.
fn run_command(prog: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(prog)
        .args(args)
        .output()
        .map_err(|e| format!("failed to execute {} {:?}: {}", prog, args, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{} {:?} failed: {}", prog, args, stderr));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| format!("invalid utf-8 output from {} {:?}: {}", prog, args, e))
}

/// Runs a command and returns stdout as raw bytes.
fn run_command_raw(prog: &str, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new(prog)
        .args(args)
        .output()
        .map_err(|e| format!("failed to execute {} {:?}: {}", prog, args, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{} {:?} failed: {}", prog, args, stderr));
    }

    Ok(output.stdout)
}

/// Runs a command with stdin data and returns stdout as a string.
fn run_command_with_stdin(prog: &str, args: &[&str], stdin: &[u8]) -> Result<String, String> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new(prog)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {} {:?}: {}", prog, args, e))?;

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin)
        .map_err(|e| format!("failed to write stdin to {} {:?}: {}", prog, args, e))?;

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for {} {:?}: {}", prog, args, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{} {:?} failed: {}", prog, args, stderr));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| format!("invalid utf-8 output from {} {:?}: {}", prog, args, e))
}
