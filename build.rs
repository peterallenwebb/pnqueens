//! Records the binary's source commit in it: the last commit that touched
//! the sources it is built from (SOURCES below), so that commits to tools
//! or documents leave the binary, its hash and its recorded commit
//! unchanged; "-dirty" when those sources have uncommitted changes. The
//! PNQUEENS_COMMIT environment variable overrides it (tools/ship.sh computes
//! the same value on the operator's machine, since a box has no history),
//! and "unknown" is recorded without git. With the `emulate` feature, also
//! builds the enumerate kernel as host C++ for the kernel emulation test
//! (kernel/emulate.cpp, src/gpu/emulate.rs).
use std::process::Command;

/// What the binary is built from. tools/ship.sh and tools/fleet.py list the same.
const SOURCES: [&str; 6] = ["src", "kernel", "Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml"];

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=PNQUEENS_COMMIT");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=kernel");
    let commit = std::env::var("PNQUEENS_COMMIT").ok().filter(|c| !c.is_empty()).unwrap_or_else(|| {
        let last = git(&[&["log", "-1", "--format=%h", "--abbrev=12", "--"], &SOURCES[..]].concat());
        let dirty = git(&[&["status", "--porcelain", "--"], &SOURCES[..]].concat());
        match (last, dirty) {
            (Some(hash), Some(status)) if status.is_empty() => hash,
            (Some(hash), _) => format!("{hash}-dirty"),
            _ => "unknown".to_string(),
        }
    });
    println!("cargo:rustc-env=PNQUEENS_COMMIT={commit}");

    #[cfg(feature = "emulate")]
    cc::Build::new().cpp(true).std("c++17").include("kernel").file("kernel/emulate.cpp").compile("nq_emulate");
}
