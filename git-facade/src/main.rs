//! Git commit vanity hash solver CLI.

use std::time::Instant;

use clap::Parser;

use git_facade::commit::parse_git_commit_object;
use git_facade::digest::hex_encode_digest;
use git_facade::digest::HexObjectDigest;
use git_facade::git;
use git_facade::solver::concurrent::ConcurrentSolver;
use git_facade::solver::gpu::GpuSolver;
use git_facade::solver::singlethreaded::SingleThreadedSolver;
use git_facade::solver::template::prepare_template;
use git_facade::solver::{CommitObject, DigestPrefixSolver};

/// CLI arguments.
#[derive(Parser)]
#[command(name = "git-facade", about = "Git commit vanity hash solver")]
struct Cli {
    /// Also update the current HEAD revision.
    #[arg(
        long,
        default_value_t = false,
        help = "Also update the current HEAD revision"
    )]
    update_ref: bool,

    /// A hex prefix to find a collision for.
    #[arg(
        long,
        default_value = "c0ffee",
        help = "A hex prefix to find a collision for"
    )]
    prefix: String,

    /// The solver to use for brute-forcing.
    #[arg(
        long,
        default_value = "concurrent",
        help = "The solver to use for brute-forcing"
    )]
    solver: String,

    /// Apply an explicit salt value instead of brute-forcing.
    #[arg(
        long,
        help = "Use an explicit salt value (1-16 hex digits) instead of brute-forcing"
    )]
    salt: Option<String>,

    /// Brute-force a matching salt and print only the salt value.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with = "salt",
        conflicts_with = "update_ref",
        help = "Find a matching salt and print only the salt value without writing or updating refs"
    )]
    salt_only: bool,
}

fn main() {
    let cli = Cli::parse();

    let prefix = hex_string_to_bytes(&cli.prefix).unwrap_or_else(|e| {
        eprintln!("invalid prefix {:?}: {}", cli.prefix, e);
        std::process::exit(1);
    });

    let hash_digest = git::get_head_digest().unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let contents = git::get_commit_contents(&hash_digest).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let obj = parse_git_commit_object(&contents).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let tpl = prepare_template(&obj).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let start = Instant::now();
    let solution = if let Some(salt_text) = cli.salt.as_deref() {
        let salt = parse_salt_value(salt_text).unwrap_or_else(|e| {
            eprintln!("invalid salt {:?}: {}", salt_text, e);
            std::process::exit(1);
        });
        apply_explicit_salt(&tpl, salt, &prefix).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        })
    } else {
        let solver: Box<dyn DigestPrefixSolver> = get_solver(&cli.solver).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

        solver.solve(&tpl, &prefix).unwrap_or_else(|e| {
            eprintln!("error: cannot find prefix collision: {}", e);
            std::process::exit(1);
        })
    };
    let duration = start.elapsed();

    if cli.salt_only {
        eprintln!(
            "found digest {} with salt {:016x} in {:.2}s",
            solution.hash,
            solution.salt,
            duration.as_secs_f64()
        );
        println!("{:016x}", solution.salt);
        return;
    }

    eprintln!(
        "found digest {} with salt {:016x} in {:.2}s",
        solution.hash,
        solution.salt,
        duration.as_secs_f64()
    );

    write_commit_object(&solution.payload, &solution.hash).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    println!("{}", solution.hash);

    if cli.update_ref {
        let reference = "HEAD";
        eprintln!("Updating {:?} to {:?}", reference, solution.hash.as_str());
        git::update_reference(reference, solution.hash.as_str()).unwrap_or_else(|e| {
            eprintln!(
                "error: failed to update branch/ref {:?} to object {:?}: {}",
                reference,
                solution.hash.as_str(),
                e
            );
            std::process::exit(1);
        });
    }
}

fn apply_explicit_salt(
    template: &git_facade::solver::template::ObjectTemplate,
    salt: u64,
    prefix: &[u8],
) -> Result<CommitObject, String> {
    let mut tpl = template.clone();
    tpl.set_salt(salt);
    let digest = tpl.sum();

    if !prefix.is_empty() && digest.0[..prefix.len()] != *prefix {
        return Err(format!(
            "provided salt {:016x} produced digest {} which does not match requested prefix {}",
            salt,
            hex_encode_digest(&digest),
            hex::encode(prefix)
        ));
    }

    Ok(CommitObject {
        salt,
        raw: tpl.bytes.clone(),
        payload: tpl.payload().to_vec(),
        hash: hex_encode_digest(&digest),
    })
}

/// Looks up a solver by name.
fn get_solver(name: &str) -> Result<Box<dyn DigestPrefixSolver>, String> {
    match name {
        "concurrent" => Ok(Box::new(ConcurrentSolver::new())),
        "gpu" => Ok(Box::new(GpuSolver::new()?)),
        "singlethreaded" => Ok(Box::new(SingleThreadedSolver::new())),
        _ => Err(format!(
            "unknown solver {:?}, available: concurrent, gpu, singlethreaded",
            name
        )),
    }
}

/// Writes a commit object to the git store and verifies the hash.
fn write_commit_object(payload: &[u8], expected_hash: &HexObjectDigest) -> Result<(), String> {
    let written_digest = git::write_object("commit", payload)?;
    if written_digest != *expected_hash {
        return Err(format!(
            "expected and written git commit object hash don't match: {:?} != {:?}",
            written_digest, expected_hash
        ));
    }
    Ok(())
}

/// Converts a hex string to bytes (e.g. "c0ffee" -> [0xc0, 0xff, 0xee]).
fn hex_string_to_bytes(s: &str) -> Result<Vec<u8>, String> {
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "odd length hex encoded bytes: len({}) = {}",
            s,
            bytes.len()
        ));
    }

    let mut result = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let upper = hex_rune_to_byte(bytes[i])?;
        let lower = hex_rune_to_byte(bytes[i + 1])?;
        result.push((upper << 4) | lower);
    }
    Ok(result)
}

fn parse_salt_value(s: &str) -> Result<u64, String> {
    if s.is_empty() {
        return Err("salt must not be empty".to_string());
    }
    if s.len() > 16 {
        return Err(format!(
            "salt must be at most 16 hex digits, got {}",
            s.len()
        ));
    }
    if !s.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("salt must contain only hex digits".to_string());
    }

    u64::from_str_radix(s, 16).map_err(|e| format!("cannot parse salt as hex u64: {}", e))
}

/// Converts a single hex ASCII byte to its numeric value.
fn hex_rune_to_byte(r: u8) -> Result<u8, String> {
    match r {
        b'0'..=b'9' => Ok(r - b'0'),
        b'a'..=b'f' => Ok(r - b'a' + 10),
        _ => Err(format!(
            "invalid hex rune, expected in [0-9a-f] but was {:?}",
            char::from(r)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_string_to_bytes_valid() {
        assert_eq!(
            hex_string_to_bytes("c0ffee").unwrap(),
            vec![0xc0, 0xff, 0xee]
        );
        assert_eq!(hex_string_to_bytes("00").unwrap(), vec![0x00]);
        assert_eq!(hex_string_to_bytes("ff").unwrap(), vec![0xff]);
        assert_eq!(hex_string_to_bytes("cafe").unwrap(), vec![0xca, 0xfe]);
        assert_eq!(
            hex_string_to_bytes("deadbeef").unwrap(),
            vec![0xde, 0xad, 0xbe, 0xef]
        );
    }

    #[test]
    fn test_hex_string_to_bytes_odd_length() {
        assert!(hex_string_to_bytes("c0f").is_err());
        assert!(hex_string_to_bytes("a").is_err());
    }

    #[test]
    fn test_hex_string_to_bytes_invalid_chars() {
        assert!(hex_string_to_bytes("zz").is_err());
        assert!(hex_string_to_bytes("GG").is_err());
        assert!(hex_string_to_bytes("c0FFEE").is_err());
    }

    #[test]
    fn test_hex_string_to_bytes_empty() {
        assert_eq!(hex_string_to_bytes("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_parse_salt_value_valid() {
        assert_eq!(parse_salt_value("0").unwrap(), 0);
        assert_eq!(parse_salt_value("1a2b").unwrap(), 0x1a2b);
        assert_eq!(
            parse_salt_value("0123456789abcdef").unwrap(),
            0x0123_4567_89ab_cdef
        );
        assert_eq!(parse_salt_value("ABCDEF").unwrap(), 0xabcdef);
    }

    #[test]
    fn test_parse_salt_value_invalid() {
        assert!(parse_salt_value("").is_err());
        assert!(parse_salt_value("xyz").is_err());
        assert!(parse_salt_value("0123456789abcdef0").is_err());
    }
}
