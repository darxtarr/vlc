//! CUDA acceleration for VLC operations
//!
//! This module implements CUDA kernels and host orchestration
//! to accelerate the core VLC operations on NVIDIA GPUs.
//!
//! Targeting sm_53+ (Jetson Nano through RTX 40xx)

pub mod context;
pub mod ops;

pub use context::{CudaContext, CudaError};
pub use ops::CudaOps;
