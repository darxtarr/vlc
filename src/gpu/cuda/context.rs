// ! CUDA context setup and management
//!
//! Handles CUDA device initialization and kernel loading for VLC operations.
//! This module provides a thin wrapper around cudarc for boutique simplicity.

use cudarc::driver::*;
use std::sync::Arc;

/// GPU context for CUDA VLC operations
pub struct CudaContext {
    pub device: Arc<CudaDevice>,
    pub stream: CudaStream,

    // Loaded kernels
    pub assign_kernel: CudaFunction,
    pub reduce_kernel: CudaFunction,
    pub update_kernel: CudaFunction,
}

impl CudaContext {
    /// Initialize CUDA context with compiled kernels
    ///
    /// Loads PTX kernels compiled at build time via build.rs
    pub fn new(device_id: usize) -> Result<Self, DriverError> {
        // Initialize CUDA driver
        cudarc::driver::result::init()?;

        // Get device
        let device = CudaDevice::new(device_id)?;

        // Create stream for async operations
        let stream = device.fork_default_stream()?;

        // Load PTX kernels (compiled by build.rs)
        let ptx = include_str!("kernels.ptx");

        // Load all kernels from the PTX module
        device.load_ptx(
            ptx.into(),
            "vlc_kernels",
            &["assign_kernel", "reduce_kernel", "update_kernel"],
        )?;

        // Get kernel functions
        let assign_kernel = device.get_func("vlc_kernels", "assign_kernel")
            .expect("Failed to load assign_kernel");
        let reduce_kernel = device.get_func("vlc_kernels", "reduce_kernel")
            .expect("Failed to load reduce_kernel");
        let update_kernel = device.get_func("vlc_kernels", "update_kernel")
            .expect("Failed to load update_kernel");

        Ok(Self {
            device,
            stream,
            assign_kernel,
            reduce_kernel,
            update_kernel,
        })
    }

    /// Synchronize stream (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.device.synchronize()
    }
}

/// CUDA-specific error type for VLC operations
#[derive(Debug)]
pub enum CudaError {
    Driver(DriverError),
    InvalidDimensions { n: usize, m: usize, d: usize },
}

impl From<DriverError> for CudaError {
    fn from(err: DriverError) -> Self {
        CudaError::Driver(err)
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Driver(e) => write!(f, "CUDA driver error: {:?}", e),
            CudaError::InvalidDimensions { n, m, d } => {
                write!(f, "Invalid dimensions: n={}, m={}, d={}", n, m, d)
            },
        }
    }
}

impl std::error::Error for CudaError {}
