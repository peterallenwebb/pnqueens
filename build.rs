//! Records the source commit in the binary (PNQUEENS_COMMIT: the environment
//! variable of that name if set, else `git rev-parse HEAD` with "-dirty" when
//! the tree has uncommitted changes, else "unknown"), and, with the `emulate`
//! feature, builds the enumerate kernel as host C++ for the kernel emulation
//! test (kernel/emulate.cpp, src/gpu/emulate.rs).
use std::process::Command;

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
        match (git(&["rev-parse", "--short=12", "HEAD"]), git(&["status", "--porcelain"])) {
            (Some(hash), Some(status)) if status.is_empty() => hash,
            (Some(hash), _) => format!("{hash}-dirty"),
            _ => "unknown".to_string(),
        }
    });
    println!("cargo:rustc-env=PNQUEENS_COMMIT={commit}");

    #[cfg(feature = "emulate")]
    cc::Build::new().cpp(true).std("c++17").include("kernel").file("kernel/emulate.cpp").compile("nq_emulate");
}
