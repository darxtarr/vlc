//! Kernel utilities and constants
//!
//! This module contains shared utilities for the GPU kernels.

/// Workgroup size used across all kernels
pub const WORKGROUP_SIZE: u32 = 256;

/// Maximum step size for anchor updates (stability)
pub const MAX_STEP_SIZE: f32 = 0.1;

/// Small epsilon for numerical stability
pub const EPSILON: f32 = 1e-6;