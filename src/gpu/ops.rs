//! GPU operations implementing the VLC algorithms
//!
//! High-level wrappers around the compute kernels with proper buffer management
//! and synchronization. This module represents the M2 GPU acceleration layer.
//!
//! ## Architecture Overview
//!
//! The GPU operations follow a three-stage pipeline:
//! 1. **Assign**: Points → nearest anchors (massively parallel)
//! 2. **Reduce**: Compute robust statistics per anchor (workgroup cooperation)
//! 3. **Update**: Move anchors toward robust means (temperature-scaled)
//!
//! ## Buffer Management Strategy
//!
//! Buffers are allocated once and reused across iterations for optimal performance.
//! The GpuOps struct maintains persistent buffers that grow as needed but never shrink,
//! following WGPU best practices for compute workloads.

use std::sync::Arc;
use half::f16;
use wgpu::{Buffer, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding};

use crate::types::{AnchorSet, Assignments};
use super::context::GpuContext;

/// GPU-accelerated operations for VLC
pub struct GpuOps {
    context: Arc<GpuContext>,

    // Persistent buffers for reuse
    points_buffer: Option<Buffer>,
    anchors_buffer: Option<Buffer>,
    assigns_buffer: Option<Buffer>,
    stats_buffer: Option<Buffer>,
    jacobians_buffer: Option<Buffer>,
    momentum_buffer: Option<Buffer>,
    params_buffer: Buffer,

    // Current buffer sizes
    current_n: usize,
    current_m: usize,
    current_d: usize,
}

impl GpuOps {
    pub fn new(context: Arc<GpuContext>) -> Self {
        // Create uniform buffer for parameters (will be reused)
        let params_buffer = context.create_uniform_buffer(64); // generous size for params

        Self {
            context,
            points_buffer: None,
            anchors_buffer: None,
            assigns_buffer: None,
            stats_buffer: None,
            jacobians_buffer: None,
            momentum_buffer: None,
            params_buffer,
            current_n: 0,
            current_m: 0,
            current_d: 0,
        }
    }

    /// Ensure buffers are allocated and sized correctly
    fn ensure_buffers(&mut self, n: usize, m: usize, d: usize) {
        if self.current_n != n || self.current_m != m || self.current_d != d {
            self.allocate_buffers(n, m, d);
            self.current_n = n;
            self.current_m = m;
            self.current_d = d;
        }
    }

    fn allocate_buffers(&mut self, n: usize, m: usize, d: usize) {
        // Points buffer [n × d] f32
        let points_size = (n * d * std::mem::size_of::<f32>()) as u64;
        self.points_buffer = Some(self.context.create_storage_buffer(points_size));

        // Anchors buffer [m × d] f32
        let anchors_size = (m * d * std::mem::size_of::<f32>()) as u64;
        self.anchors_buffer = Some(self.context.create_storage_buffer(anchors_size));

        // Assignments buffer [n] u32
        let assigns_size = (n * std::mem::size_of::<u32>()) as u64;
        self.assigns_buffer = Some(self.context.create_storage_buffer(assigns_size));

        // Stats buffer [m × (d+2)] f32
        let stats_size = (m * (d + 2) * std::mem::size_of::<f32>()) as u64;
        self.stats_buffer = Some(self.context.create_storage_buffer(stats_size));

        // Jacobians buffer [m × d] f32
        let jacobians_size = anchors_size;
        self.jacobians_buffer = Some(self.context.create_storage_buffer(jacobians_size));

        // Momentum buffer [m × d] f32
        let momentum_size = anchors_size;
        self.momentum_buffer = Some(self.context.create_storage_buffer(momentum_size));
    }

    /// Assign points to nearest anchors (GPU implementation)
    ///
    /// Follows WGPU 26.0 best practices for compute operations.
    pub async fn assign_points(
        &mut self,
        points: &[f16],
        anchors: &AnchorSet,
        n: usize,
        d: usize,
    ) -> Result<Assignments, Box<dyn std::error::Error>> {
        self.ensure_buffers(n, anchors.m, d);

        // Convert f16 points to f32 for GPU
        let points_f32: Vec<f32> = points.iter().map(|&x| x.to_f32()).collect();
        let anchors_f32: Vec<f32> = anchors.anchors.iter().map(|&x| x.to_f32()).collect();

        // Upload data to GPU
        self.context.queue.write_buffer(
            self.points_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&points_f32),
        );

        self.context.queue.write_buffer(
            self.anchors_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&anchors_f32),
        );

        // Upload parameters
        let params = AssignParams {
            n: n as u32,
            m: anchors.m as u32,
            d: d as u32,
            _padding: 0,
        };
        self.context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        // Create bind groups
        let storage_bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Assign Storage Bind Group"),
            layout: &self.context.assign_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: self.points_buffer.as_ref().unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: self.anchors_buffer.as_ref().unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: self.assigns_buffer.as_ref().unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let uniform_bind_group = self.context.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Assign Uniform Bind Group"),
            layout: &self.context.assign_pipeline.get_bind_group_layout(1),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &self.params_buffer,
                    offset: 0,
                    size: std::num::NonZero::new(std::mem::size_of::<AssignParams>() as u64),
                }),
            }],
        });

        // Dispatch compute
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&self.context.assign_pipeline);
            compute_pass.set_bind_group(0, &storage_bind_group, &[]);
            compute_pass.set_bind_group(1, &uniform_bind_group, &[]);

            let workgroups = (n + 255) / 256; // Round up division
            compute_pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        // Submit and wait
        self.context.queue.submit([encoder.finish()]);
        let _ = self.context.device.poll(wgpu::PollType::Wait);

        // Read back results
        let staging_buffer = self.context.create_staging_buffer(
            (n * std::mem::size_of::<u32>()) as u64
        );

        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            self.assigns_buffer.as_ref().unwrap(),
            0,
            &staging_buffer,
            0,
            (n * std::mem::size_of::<u32>()) as u64,
        );
        self.context.queue.submit([encoder.finish()]);

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        let _ = self.context.device.poll(wgpu::PollType::Wait);
        receiver.receive().await.unwrap()?;

        let data = buffer_slice.get_mapped_range();
        let assignments: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(Assignments {
            assign: assignments,
            residuals: None,
            n,
            d_r: 0,
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AssignParams {
    n: u32,
    m: u32,
    d: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    n: u32,
    m: u32,
    d: u32,
    huber_threshold: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateParams {
    m: u32,
    d: u32,
    temperature: f32,
    learning_rate: f32,
    momentum: f32,
    enable_jacobians: u32,
    _padding: [u32; 2],
}