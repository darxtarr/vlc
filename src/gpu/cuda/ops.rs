//! CUDA GPU operations implementing the VLC algorithms
//!
//! High-level wrappers around CUDA kernels with proper buffer management.
//! Follows the same pattern as the WGPU implementation for consistency.
//!
//! ## Architecture Overview
//!
//! The GPU operations follow a three-stage pipeline:
//! 1. **Assign**: Points to nearest anchors (massively parallel)
//! 2. **Reduce**: Compute robust statistics per anchor (workgroup cooperation)
//! 3. **Update**: Move anchors toward robust means (temperature-scaled)

use cudarc::driver::*;
use half::f16;
use std::sync::Arc;

use crate::types::{AnchorSet, Assignments};
use super::context::{CudaContext, CudaError};

/// CUDA-accelerated operations for VLC
pub struct CudaOps {
    context: Arc<CudaContext>,

    // Persistent buffers for reuse
    points_buffer: Option<CudaSlice<f32>>,
    anchors_buffer: Option<CudaSlice<f32>>,
    assigns_buffer: Option<CudaSlice<u32>>,
    stats_buffer: Option<CudaSlice<f32>>,
    jacobians_buffer: Option<CudaSlice<f32>>,
    momentum_buffer: Option<CudaSlice<f32>>,

    // Current buffer sizes
    current_n: usize,
    current_m: usize,
    current_d: usize,
}

impl CudaOps {
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            points_buffer: None,
            anchors_buffer: None,
            assigns_buffer: None,
            stats_buffer: None,
            jacobians_buffer: None,
            momentum_buffer: None,
            current_n: 0,
            current_m: 0,
            current_d: 0,
        }
    }

    /// Ensure buffers are allocated and sized correctly
    fn ensure_buffers(&mut self, n: usize, m: usize, d: usize) -> Result<(), CudaError> {
        if self.current_n != n || self.current_m != m || self.current_d != d {
            self.allocate_buffers(n, m, d)?;
            self.current_n = n;
            self.current_m = m;
            self.current_d = d;
        }
        Ok(())
    }

    fn allocate_buffers(&mut self, n: usize, m: usize, d: usize) -> Result<(), CudaError> {
        // Allocate buffers on GPU (will fail if insufficient memory)
        self.points_buffer = Some(self.context.device.alloc_zeros::<f32>(n * d)?);
        self.anchors_buffer = Some(self.context.device.alloc_zeros::<f32>(m * d)?);
        self.assigns_buffer = Some(self.context.device.alloc_zeros::<u32>(n)?);
        self.stats_buffer = Some(self.context.device.alloc_zeros::<f32>(m * (d + 2))?);
        self.jacobians_buffer = Some(self.context.device.alloc_zeros::<f32>(m * d)?);
        self.momentum_buffer = Some(self.context.device.alloc_zeros::<f32>(m * d)?);

        Ok(())
    }

    /// Assign points to nearest anchors (CUDA implementation)
    pub fn assign_points(
        &mut self,
        points: &[f16],
        anchors: &AnchorSet,
        n: usize,
        d: usize,
    ) -> Result<Assignments, CudaError> {
        self.ensure_buffers(n, anchors.m, d)?;

        // Convert f16 to f32
        let points_f32: Vec<f32> = points.iter().map(|&x| x.to_f32()).collect();
        let anchors_f32: Vec<f32> = anchors.anchors.iter().map(|&x| x.to_f32()).collect();

        // Upload data to GPU
        self.context.device.htod_sync_copy_into(
            &points_f32,
            self.points_buffer.as_mut().unwrap(),
        )?;

        self.context.device.htod_sync_copy_into(
            &anchors_f32,
            self.anchors_buffer.as_mut().unwrap(),
        )?;

        // Prepare kernel parameters
        let params = AssignParams {
            n: n as u32,
            m: anchors.m as u32,
            d: d as u32,
        };

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.context.assign_kernel.clone().launch(
                cfg,
                (
                    self.points_buffer.as_ref().unwrap(),
                    self.anchors_buffer.as_ref().unwrap(),
                    self.assigns_buffer.as_mut().unwrap(),
                    params,
                ),
            )?;
        }

        // Synchronize
        self.context.synchronize()?;

        // Read back results
        let mut assignments = vec![0u32; n];
        self.context.device.dtoh_sync_copy_into(
            self.assigns_buffer.as_ref().unwrap(),
            &mut assignments,
        )?;

        Ok(Assignments {
            assign: assignments,
            residuals: None,
            n,
            d_r: 0,
        })
    }

    /// Compute robust statistics for each anchor (CUDA implementation)
    pub fn reduce_stats(
        &mut self,
        points: &[f16],
        assignments: &Assignments,
        _anchors: &AnchorSet,
        n: usize,
        m: usize,
        d: usize,
    ) -> Result<Vec<crate::types::AnchorStats>, CudaError> {
        self.ensure_buffers(n, m, d)?;

        // Convert f16 to f32
        let points_f32: Vec<f32> = points.iter().map(|&x| x.to_f32()).collect();

        // Upload data
        self.context.device.htod_sync_copy_into(
            &points_f32,
            self.points_buffer.as_mut().unwrap(),
        )?;

        self.context.device.htod_sync_copy_into(
            &assignments.assign,
            self.assigns_buffer.as_mut().unwrap(),
        )?;

        // Prepare kernel parameters
        let params = ReduceParams {
            n: n as u32,
            m: m as u32,
            d: d as u32,
            huber_threshold: 1.0,
        };

        // Launch kernel: one block per anchor
        // Shared memory: 256 floats + 1 uint32 = 256*4 + 4 = 1028 bytes
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (256 * std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32,
        };

        unsafe {
            self.context.reduce_kernel.clone().launch(
                cfg,
                (
                    self.points_buffer.as_ref().unwrap(),
                    self.assigns_buffer.as_ref().unwrap(),
                    self.stats_buffer.as_mut().unwrap(),
                    params,
                ),
            )?;
        }

        // Synchronize
        self.context.synchronize()?;

        // Read back stats
        let mut stats_data = vec![0.0f32; m * (d + 2)];
        self.context.device.dtoh_sync_copy_into(
            self.stats_buffer.as_ref().unwrap(),
            &mut stats_data,
        )?;

        // Convert to AnchorStats
        let mut result = Vec::with_capacity(m);
        for anchor_idx in 0..m {
            let base = anchor_idx * (d + 2);
            let mean = stats_data[base..base + d].to_vec();
            let count = stats_data[base + d] as usize;
            let variance = vec![stats_data[base + d + 1]; d];

            result.push(crate::types::AnchorStats {
                mean,
                count,
                variance,
            });
        }

        Ok(result)
    }

    /// Update anchor positions (CUDA implementation)
    pub fn update_anchors(
        &mut self,
        anchors: &mut AnchorSet,
        stats: &[crate::types::AnchorStats],
        temperature: f32,
        learning_rate: f32,
    ) -> Result<(), CudaError> {
        let m = anchors.m;
        let d = anchors.d;

        self.ensure_buffers(1, m, d)?;

        // Convert anchors to f32
        let anchors_f32: Vec<f32> = anchors.anchors.iter().map(|&x| x.to_f32()).collect();

        // Convert stats to flat buffer [m x (d+2)]
        let mut stats_f32 = Vec::with_capacity(m * (d + 2));
        for stat in stats {
            stats_f32.extend(&stat.mean);
            stats_f32.push(stat.count as f32);
            let avg_variance = stat.variance.iter().sum::<f32>() / d as f32;
            stats_f32.push(avg_variance);
        }

        // Upload data
        self.context.device.htod_sync_copy_into(
            &anchors_f32,
            self.anchors_buffer.as_mut().unwrap(),
        )?;

        self.context.device.htod_sync_copy_into(
            &stats_f32,
            self.stats_buffer.as_mut().unwrap(),
        )?;

        // Prepare kernel parameters
        let params = UpdateParams {
            m: m as u32,
            d: d as u32,
            temperature,
            learning_rate,
            momentum: 0.0,
            enable_jacobians: 0,
        };

        // Launch kernel: (m * d) threads total
        let total_threads = (m * d) as u32;
        let cfg = LaunchConfig {
            grid_dim: ((total_threads + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.context.update_kernel.clone().launch(
                cfg,
                (
                    self.anchors_buffer.as_mut().unwrap(),
                    self.stats_buffer.as_ref().unwrap(),
                    self.jacobians_buffer.as_mut().unwrap(),
                    self.momentum_buffer.as_mut().unwrap(),
                    params,
                ),
            )?;
        }

        // Synchronize
        self.context.synchronize()?;

        // Read back updated anchors
        let mut updated_anchors_f32 = vec![0.0f32; m * d];
        self.context.device.dtoh_sync_copy_into(
            self.anchors_buffer.as_ref().unwrap(),
            &mut updated_anchors_f32,
        )?;

        // Convert back to f16
        anchors.anchors = updated_anchors_f32
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();

        Ok(())
    }
}

// Kernel parameter structs (must match CUDA kernels)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct AssignParams {
    n: u32,
    m: u32,
    d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ReduceParams {
    n: u32,
    m: u32,
    d: u32,
    huber_threshold: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UpdateParams {
    m: u32,
    d: u32,
    temperature: f32,
    learning_rate: f32,
    momentum: f32,
    enable_jacobians: u32,
}

// Implement DeviceRepr for parameter structs so cudarc can pass them
unsafe impl DeviceRepr for AssignParams {}
unsafe impl DeviceRepr for ReduceParams {}
unsafe impl DeviceRepr for UpdateParams {}
