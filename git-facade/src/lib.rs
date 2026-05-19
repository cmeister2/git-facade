//! Git commit vanity hash solver library.
//!
//! Finds a salt value that makes a git commit's SHA1 hash start with a desired
//! hex prefix (e.g. `c0ffee`, `cafe`, `facade`).

pub mod commit;
pub mod digest;
pub mod git;
pub mod solver;
