use std::process::Command;

#[cfg(feature = "gpu-cuda")]
fn build_cuda_kernels() {
    println!("cargo:rerun-if-changed=src/gpu/cuda/kernels.cu");

    // Compile CUDA kernels to PTX
    // PTX is forward-compatible, so we target compute_86 (RTX 4080)
    // This will work on sm_86 and newer architectures
    // For Jetson Nano (sm_53), we'd need a separate build target
    let status = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-O3",
            "--gpu-architecture=compute_86", // Ada Lovelace (RTX 4080)
            "src/gpu/cuda/kernels.cu",
            "-o",
            "src/gpu/cuda/kernels.ptx",
        ])
        .status()
        .expect("Failed to run nvcc - make sure CUDA toolkit is installed");

    if !status.success() {
        panic!("nvcc compilation failed - check CUDA toolkit installation");
    }
    println!("cargo:warning=CUDA kernels compiled successfully");
}

fn main() {
    #[cfg(feature = "gpu-cuda")]
    build_cuda_kernels();
}
