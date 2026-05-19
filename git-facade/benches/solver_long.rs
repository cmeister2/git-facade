//! Long-prefix solver benchmarks.
//!
//! Separate from the fast benchmarks because these take significantly longer
//! per iteration (seconds rather than microseconds).

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

use git_facade::commit::parse_git_commit_object;
use git_facade::solver::concurrent::ConcurrentSolver;
use git_facade::solver::gpu::GpuSolver;
use git_facade::solver::template::prepare_template;
use git_facade::solver::DigestPrefixSolver;

/// Test fixture: a raw commit object header and body.
const RAW_HEADER_AND_BODY_OBJECT: &str = "tree e57181f20b062532907436169bb5823b6af2f099\n\
    author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
    committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
    \n\
    Initial commit\n\
    36abde0100000000";

/// 3-byte prefix (6 hex chars): expected ~2^24 candidates on average.
const LONG_PREFIX: [u8; 3] = [0xfa, 0xca, 0xde];

/// Benchmarks the concurrent solver finding a 3-byte prefix.
fn bench_concurrent_long(c: &mut Criterion) {
    let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
    let tpl = prepare_template(&obj).unwrap();

    c.bench_function("concurrent_3byte", |b| {
        b.iter(|| {
            let solver = ConcurrentSolver::new();
            solver.solve(&tpl, &LONG_PREFIX).unwrap();
        });
    });
}

/// Benchmarks the GPU solver finding a 3-byte prefix.
fn bench_gpu_long(c: &mut Criterion) {
    let gpu_solver = match GpuSolver::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipping GPU benchmark (no adapter): {}", e);
            return;
        }
    };

    let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
    let tpl = prepare_template(&obj).unwrap();

    c.bench_function("gpu_3byte", |b| {
        b.iter(|| {
            gpu_solver.solve(&tpl, &LONG_PREFIX).unwrap();
        });
    });
}

#[allow(missing_docs, clippy::missing_docs_in_private_items)]
mod group {
    use super::*;
    criterion_group! {
        name = long_benches;
        config = Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(60));
        targets = bench_concurrent_long, bench_gpu_long,
    }
}

criterion_main!(group::long_benches);
