//! With the `emulate` feature, builds the enumerate kernel as host C++ for
//! the kernel emulation test (kernel/emulate.cpp, src/gpu/emulate.rs).
fn main() {
    println!("cargo:rerun-if-changed=kernel/emulate.cpp");
    println!("cargo:rerun-if-changed=kernel/enumerate.cu");
    #[cfg(feature = "emulate")]
    cc::Build::new().cpp(true).std("c++17").include("kernel").file("kernel/emulate.cpp").compile("nq_emulate");
}
