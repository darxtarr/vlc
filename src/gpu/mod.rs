//! GPU acceleration for VLC operations using WGPU
//!
//! This module implements the compute shaders and host orchestration
//! to accelerate the core VLC operations on GPU.

pub mod context;
pub mod kernels;
pub mod ops;

pub use context::GpuContext;
pub use ops::GpuOps;