//! GPU acceleration for VLC operations
//!
//! This module provides GPU acceleration through feature-gated backends:
//! - `gpu-wgpu`: Portable GPU acceleration via WGPU/Vulkan (default)
//! - `gpu-cuda`: NVIDIA-specific acceleration via CUDA
//!
//! The backend is selected at compile time via Cargo features.

// WGPU backend (default)
#[cfg(feature = "gpu-wgpu")]
pub mod context;
#[cfg(feature = "gpu-wgpu")]
pub mod kernels;
#[cfg(feature = "gpu-wgpu")]
pub mod ops;

#[cfg(feature = "gpu-wgpu")]
pub use context::GpuContext;
#[cfg(feature = "gpu-wgpu")]
pub use ops::GpuOps;

// CUDA backend
#[cfg(feature = "gpu-cuda")]
pub mod cuda;

#[cfg(feature = "gpu-cuda")]
pub use cuda::{CudaContext, CudaOps, CudaError};