//! GPU prefix timing tester.

use std::time::{Duration, Instant};

use clap::Parser;
use git_facade::commit::parse_git_commit_object;
use git_facade::digest::hex_encode_digest;
use git_facade::git;
use git_facade::solver::has_prefix;
use git_facade::solver::template::{prepare_template, ObjectTemplate};
use wgpu_sha1::{GpuSha1, GpuTemplate, DEFAULT_BATCH_SIZE};

/// Command-line arguments for the GPU prefix tester.
#[derive(Parser)]
#[command(
    name = "gpu-prefix-tester",
    about = "Time the GPU solver on 2-, 3-, and 4-byte prefixes"
)]
struct Cli {
    /// Two-byte hex prefix to test.
    #[arg(long = "prefix-2", default_value = "8870", value_name = "HEX")]
    prefix_2: String,

    /// Three-byte hex prefix to test.
    #[arg(long = "prefix-3", default_value = "facade", value_name = "HEX")]
    prefix_3: String,

    /// Four-byte hex prefix to test.
    #[arg(long = "prefix-4", default_value = "00000000", value_name = "HEX")]
    prefix_4: String,

    /// Number of times to run each prefix test.
    #[arg(long, default_value_t = 1)]
    repeats: u32,

    /// Number of candidate salts per GPU dispatch.
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: u32,
}

/// A parsed prefix test case.
struct PrefixCase {
    /// Human-readable case label.
    label: &'static str,
    /// Prefix bytes to search for.
    prefix: Vec<u8>,
    /// Original hex prefix string.
    hex: String,
}

/// Result from one timed GPU search.
struct TimedRun {
    /// Wall-clock time spent searching.
    duration: Duration,
    /// Matching salt value.
    salt: u64,
    /// Matching SHA1 digest as lowercase hex.
    hash: String,
    /// Number of GPU dispatches submitted.
    batches: u64,
    /// Number of candidate invocations submitted to the GPU.
    dispatched_candidates: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {}", error);
        std::process::exit(1);
    }
}

/// Runs the GPU prefix tester.
fn run() -> Result<(), String> {
    let cli = Cli::parse();
    if cli.repeats == 0 {
        return Err("--repeats must be at least 1".to_string());
    }
    if cli.batch_size == 0 {
        return Err("--batch-size must be at least 1".to_string());
    }

    let cases = [
        parse_prefix_case("2-byte", &cli.prefix_2, 2)?,
        parse_prefix_case("3-byte", &cli.prefix_3, 3)?,
        parse_prefix_case("4-byte", &cli.prefix_4, 4)?,
    ];

    let template = load_head_template()?;
    let gpu_template = GpuTemplate::from_bytes(&template.bytes, template.salt_offset);
    let gpu = GpuSha1::new().map_err(|error| error.to_string())?;

    println!("batch_size={} repeats={}", cli.batch_size, cli.repeats);
    println!(
        "{:<8} {:<10} {:>5} {:>12} {:>18} {:>14} {:>14}  hash",
        "case", "prefix", "run", "time_ms", "salt", "batches", "candidates/s"
    );

    for case in cases {
        let mut durations = Vec::new();
        for run_index in 1..=cli.repeats {
            let run = time_prefix(&gpu, &gpu_template, &template, &case.prefix, cli.batch_size)?;
            durations.push(run.duration);

            let candidates_per_second =
                run.dispatched_candidates as f64 / run.duration.as_secs_f64();
            println!(
                "{:<8} {:<10} {:>5} {:>12.3} {:>18} {:>14} {:>14.0}  {}",
                case.label,
                case.hex,
                run_index,
                run.duration.as_secs_f64() * 1000.0,
                run.salt,
                run.batches,
                candidates_per_second,
                run.hash,
            );
        }

        if cli.repeats > 1 {
            print_summary(&case, &durations);
        }
    }

    Ok(())
}

/// Loads the current HEAD commit and prepares it for salt brute-forcing.
fn load_head_template() -> Result<ObjectTemplate, String> {
    let digest = git::get_head_digest()?;
    let contents = git::get_commit_contents(&digest)?;
    let object = parse_git_commit_object(&contents)?;
    prepare_template(&object)
}

/// Parses a fixed-width lowercase hex prefix test case.
fn parse_prefix_case(
    label: &'static str,
    hex: &str,
    byte_len: usize,
) -> Result<PrefixCase, String> {
    let prefix = hex_string_to_bytes(hex)?;
    if prefix.len() != byte_len {
        return Err(format!(
            "{} prefix must be exactly {} bytes / {} hex chars, got {:?}",
            label,
            byte_len,
            byte_len * 2,
            hex
        ));
    }

    Ok(PrefixCase {
        label,
        prefix,
        hex: hex.to_string(),
    })
}

/// Times one GPU prefix search and verifies the returned hash on the CPU.
fn time_prefix(
    gpu: &GpuSha1,
    gpu_template: &GpuTemplate,
    template: &ObjectTemplate,
    prefix: &[u8],
    batch_size: u32,
) -> Result<TimedRun, String> {
    let start = Instant::now();
    let mut salt_base = 0u64;
    let mut batches = 0u64;

    loop {
        let result = gpu
            .find_prefix(gpu_template, prefix, salt_base, batch_size)
            .map_err(|error| error.to_string())?;
        batches += 1;

        if let Some(found) = result {
            let duration = start.elapsed();
            let mut solved_template = template.clone();
            solved_template.set_salt(found.salt);
            let digest = solved_template.sum();
            if !has_prefix(&digest, prefix) {
                return Err(format!(
                    "GPU returned salt {} but CPU hash {} does not match prefix {}",
                    found.salt,
                    digest,
                    hex::encode(prefix)
                ));
            }

            return Ok(TimedRun {
                duration,
                salt: found.salt,
                hash: hex_encode_digest(&digest).to_string(),
                batches,
                dispatched_candidates: batches * u64::from(batch_size),
            });
        }

        salt_base = salt_base
            .checked_add(u64::from(batch_size))
            .ok_or_else(|| "exhausted u64 salt space".to_string())?;
    }
}

/// Prints aggregate timing for repeated runs of one case.
fn print_summary(case: &PrefixCase, durations: &[Duration]) {
    let total_seconds: f64 = durations.iter().map(Duration::as_secs_f64).sum();
    let average_ms = total_seconds * 1000.0 / durations.len() as f64;
    let min_ms = durations
        .iter()
        .map(Duration::as_secs_f64)
        .fold(f64::INFINITY, f64::min)
        * 1000.0;
    let max_ms = durations
        .iter()
        .map(Duration::as_secs_f64)
        .fold(0.0, f64::max)
        * 1000.0;

    println!(
        "{:<8} {:<10} {:>5} {:>12.3} {:>18} {:>14} {:>14}  min={:.3}ms max={:.3}ms",
        case.label, case.hex, "avg", average_ms, "-", "-", "-", min_ms, max_ms
    );
}

/// Converts a lowercase hex string to bytes.
fn hex_string_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    let bytes = hex.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "odd length hex encoded bytes: len({}) = {}",
            hex,
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

/// Converts one lowercase hex ASCII byte to its nibble value.
fn hex_rune_to_byte(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(format!(
            "invalid hex rune, expected in [0-9a-f] but was {:?}",
            char::from(byte)
        )),
    }
}
