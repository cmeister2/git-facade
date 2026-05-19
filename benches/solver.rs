//! Solver benchmarks.

use criterion::{criterion_group, criterion_main, Criterion};
use sha1::{Digest, Sha1};

use git_facade::commit::parse_git_commit_object;
use git_facade::solver::concurrent::ConcurrentSolver;
use git_facade::solver::singlethreaded::SingleThreadedSolver;
use git_facade::solver::template::prepare_template;
use git_facade::solver::DigestPrefixSolver;

/// Test fixture: a raw commit object header and body.
const RAW_HEADER_AND_BODY_OBJECT: &str = "tree e57181f20b062532907436169bb5823b6af2f099\n\
    author Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
    committer Thomas Richner <thomas.richner@oviva.com> 1653693519 +0200\n\
    \n\
    Initial commit\n\
    36abde0100000000";

/// Benchmarks full SHA1 digest of the entire commit object.
fn bench_sha1_full(c: &mut Criterion) {
    let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
    let tpl = prepare_template(&obj).unwrap();

    c.bench_function("sha1_full", |b| {
        b.iter(|| Sha1::digest(&tpl.bytes));
    });
}

/// Benchmarks incremental SHA1 that only hashes the tail block.
fn bench_sha1_incremental(c: &mut Criterion) {
    let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
    let tpl = prepare_template(&obj).unwrap();
    let hasher = tpl.incremental_hasher();

    c.bench_function("sha1_incremental", |b| {
        b.iter(|| hasher.sum(&tpl));
    });
}

/// Benchmarks the single-threaded solver finding a 2-byte prefix.
fn bench_solver_singlethreaded(c: &mut Criterion) {
    let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
    let tpl = prepare_template(&obj).unwrap();

    c.bench_function("solver_singlethreaded", |b| {
        b.iter(|| {
            let solver = SingleThreadedSolver::with_range(0, 4096 * 1024);
            solver.solve(&tpl, &[0x88, 0x70]).unwrap();
        });
    });
}

/// Benchmarks the concurrent solver finding a 2-byte prefix.
fn bench_solver_concurrent(c: &mut Criterion) {
    let obj = parse_git_commit_object(RAW_HEADER_AND_BODY_OBJECT.as_bytes()).unwrap();
    let tpl = prepare_template(&obj).unwrap();

    c.bench_function("solver_concurrent", |b| {
        b.iter(|| {
            let solver = ConcurrentSolver::new();
            solver.solve(&tpl, &[0x88, 0x70]).unwrap();
        });
    });
}

#[allow(missing_docs, clippy::missing_docs_in_private_items)]
mod group {
    use super::*;
    criterion_group!(
        benches,
        bench_sha1_full,
        bench_sha1_incremental,
        bench_solver_singlethreaded,
        bench_solver_concurrent,
    );
}

criterion_main!(group::benches);
