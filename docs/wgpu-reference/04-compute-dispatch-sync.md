# Compute Shader Dispatch and Synchronization - WGPU 26.0

## Overview

Compute shader dispatch involves encoding compute passes, setting up pipelines and bind groups, dispatching workgroups, and managing synchronization between CPU and GPU operations.

## Basic Compute Dispatch Pattern

```rust
use wgpu::{
    CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipeline,
    BindGroup, Device, Queue
};

fn dispatch_compute(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_groups: &[&BindGroup],
    workgroups: (u32, u32, u32),
) {
    // 1. Create command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    // 2. Begin compute pass
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None, // For profiling
        });

        // 3. Set pipeline
        compute_pass.set_pipeline(pipeline);

        // 4. Set bind groups
        for (index, bind_group) in bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(index as u32, bind_group, &[]);
        }

        // 5. Dispatch workgroups
        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    } // Compute pass ends here

    // 6. Submit commands
    queue.submit([encoder.finish()]);
}
```

## ComputePassDescriptor

```rust
use wgpu::{ComputePassDescriptor, ComputePassTimestampWrites, QuerySet};

// Basic compute pass
let basic_descriptor = ComputePassDescriptor {
    label: Some("Basic Compute"),
    timestamp_writes: None,
};

// Compute pass with profiling
let profiling_descriptor = ComputePassDescriptor {
    label: Some("Profiled Compute"),
    timestamp_writes: Some(ComputePassTimestampWrites {
        query_set: &timestamp_query_set,
        beginning_of_pass_write_index: Some(0),
        end_of_pass_write_index: Some(1),
    }),
};
```

## Workgroup Calculation

### 1D Dispatch

```rust
fn calculate_workgroups_1d(data_size: u32, workgroup_size: u32) -> u32 {
    (data_size + workgroup_size - 1) / workgroup_size // Ceiling division
}

// Example: Process 10,000 elements with workgroup size 256
let workgroups = calculate_workgroups_1d(10_000, 256); // = 40 workgroups
compute_pass.dispatch_workgroups(workgroups, 1, 1);
```

### 2D Dispatch

```rust
fn calculate_workgroups_2d(
    width: u32,
    height: u32,
    workgroup_x: u32,
    workgroup_y: u32,
) -> (u32, u32) {
    let x = (width + workgroup_x - 1) / workgroup_x;
    let y = (height + workgroup_y - 1) / workgroup_y;
    (x, y)
}

// Example: Process 1920x1080 texture with 16x16 workgroups
let (x, y) = calculate_workgroups_2d(1920, 1080, 16, 16);
compute_pass.dispatch_workgroups(x, y, 1);
```

### 3D Dispatch

```rust
fn calculate_workgroups_3d(
    dimensions: (u32, u32, u32),
    workgroup_size: (u32, u32, u32),
) -> (u32, u32, u32) {
    let x = (dimensions.0 + workgroup_size.0 - 1) / workgroup_size.0;
    let y = (dimensions.1 + workgroup_size.1 - 1) / workgroup_size.1;
    let z = (dimensions.2 + workgroup_size.2 - 1) / workgroup_size.2;
    (x, y, z)
}

// Example: Process 3D volume
let (x, y, z) = calculate_workgroups_3d((256, 256, 128), (8, 8, 4));
compute_pass.dispatch_workgroups(x, y, z);
```

## Advanced Dispatch Patterns

### Indirect Dispatch

```rust
use wgpu::BufferUsages;

// Create indirect buffer with dispatch arguments
fn create_indirect_buffer(device: &Device, args: &[u32; 3]) -> Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Indirect Dispatch Buffer"),
        size: 12, // 3 u32s = 12 bytes
        usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    // Write dispatch arguments
    buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(args));
    buffer.unmap();

    buffer
}

// Dispatch indirectly
fn dispatch_compute_indirect(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_groups: &[&BindGroup],
    indirect_buffer: &Buffer,
    offset: u64,
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(pipeline);

        for (index, bind_group) in bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(index as u32, bind_group, &[]);
        }

        compute_pass.dispatch_workgroups_indirect(indirect_buffer, offset);
    }
    queue.submit([encoder.finish()]);
}
```

### Multi-Stage Compute Pipeline

```rust
fn multi_stage_compute(
    device: &Device,
    queue: &Queue,
    pipelines: &[ComputePipeline],
    bind_groups: &[Vec<BindGroup>],
    workgroups: &[(u32, u32, u32)],
) {
    let mut encoder = device.create_command_encoder(&Default::default());

    for (stage_idx, (pipeline, stage_bind_groups, stage_workgroups)) in
        pipelines.iter().zip(bind_groups.iter()).zip(workgroups.iter()).enumerate() {

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(&format!("Compute Stage {}", stage_idx)),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);

        for (bind_idx, bind_group) in stage_bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(bind_idx as u32, bind_group, &[]);
        }

        compute_pass.dispatch_workgroups(stage_workgroups.0, stage_workgroups.1, stage_workgroups.2);
    }

    queue.submit([encoder.finish()]);
}
```

## Synchronization Strategies

### 1. Immediate Synchronization (Blocking)

```rust
fn immediate_sync_dispatch(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_groups: &[&BindGroup],
    workgroups: (u32, u32, u32),
) {
    // Dispatch compute
    dispatch_compute(device, queue, pipeline, bind_groups, workgroups);

    // Block until completion
    device.poll(wgpu::Maintain::Wait);
}
```

### 2. Async Synchronization (Non-blocking)

```rust
use futures_intrusive::channel::shared::oneshot_channel;

async fn async_compute_dispatch(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_groups: &[&BindGroup],
    workgroups: (u32, u32, u32),
) {
    // Dispatch compute
    dispatch_compute(device, queue, pipeline, bind_groups, workgroups);

    // Set up async completion callback
    let (sender, receiver) = oneshot_channel();
    queue.on_submitted_work_done(move || {
        sender.send(()).unwrap();
    });

    // Continue other work...

    // Wait for completion when needed
    receiver.receive().await.unwrap();
}
```

### 3. Pipeline Synchronization

```rust
fn pipelined_compute(
    device: &Device,
    queue: &Queue,
    data_batches: &[&[f32]],
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
) {
    let mut encoders = Vec::new();

    // Submit multiple compute operations
    for (batch_idx, batch_data) in data_batches.iter().enumerate() {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Batch {} Encoder", batch_idx)),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("Batch {} Pass", batch_idx)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            let workgroups = (batch_data.len() as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoders.push(encoder.finish());
    }

    // Submit all batches together
    queue.submit(encoders);
}
```

## Memory Barriers and Dependencies

### Buffer Dependencies

```rust
fn compute_with_buffer_dependency(
    device: &Device,
    queue: &Queue,
    pipeline1: &ComputePipeline,
    pipeline2: &ComputePipeline,
    bind_groups1: &[&BindGroup],
    bind_groups2: &[&BindGroup],
    intermediate_buffer: &Buffer,
) {
    let mut encoder = device.create_command_encoder(&Default::default());

    // First compute pass
    {
        let mut compute_pass1 = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Pass 1"),
            timestamp_writes: None,
        });

        compute_pass1.set_pipeline(pipeline1);
        for (i, bg) in bind_groups1.iter().enumerate() {
            compute_pass1.set_bind_group(i as u32, bg, &[]);
        }
        compute_pass1.dispatch_workgroups(256, 1, 1);
    }

    // Implicit memory barrier here between compute passes

    // Second compute pass that depends on first
    {
        let mut compute_pass2 = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Pass 2"),
            timestamp_writes: None,
        });

        compute_pass2.set_pipeline(pipeline2);
        for (i, bg) in bind_groups2.iter().enumerate() {
            compute_pass2.set_bind_group(i as u32, bg, &[]);
        }
        compute_pass2.dispatch_workgroups(256, 1, 1);
    }

    queue.submit([encoder.finish()]);
}
```

## Performance Profiling

### Timestamp Queries

```rust
use wgpu::{QuerySet, QuerySetDescriptor, QueryType, Features};

fn create_timestamp_query_set(device: &Device, count: u32) -> QuerySet {
    device.create_query_set(&QuerySetDescriptor {
        label: Some("Timestamp Queries"),
        ty: QueryType::Timestamp,
        count,
    })
}

fn profiled_compute_dispatch(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_groups: &[&BindGroup],
    workgroups: (u32, u32, u32),
    query_set: &QuerySet,
) {
    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Profiled Compute"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });

        compute_pass.set_pipeline(pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, bg, &[]);
        }
        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // Resolve timestamps to buffer
    let timestamp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Buffer"),
        size: 16, // 2 timestamps * 8 bytes each
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    encoder.resolve_query_set(query_set, 0..2, &timestamp_buffer, 0);
    queue.submit([encoder.finish()]);
}
```

## Error Handling

```rust
use wgpu::{Error, SubmissionIndex};

fn safe_compute_dispatch(
    device: &Device,
    queue: &Queue,
    pipeline: &ComputePipeline,
    bind_groups: &[&BindGroup],
    workgroups: (u32, u32, u32),
) -> Result<SubmissionIndex, String> {
    // Validate workgroup dimensions
    let limits = device.limits();
    if workgroups.0 > limits.max_compute_workgroups_per_dimension ||
       workgroups.1 > limits.max_compute_workgroups_per_dimension ||
       workgroups.2 > limits.max_compute_workgroups_per_dimension {
        return Err("Workgroup dimensions exceed device limits".to_string());
    }

    // Set up error scope
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(pipeline);

        for (i, bg) in bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, bg, &[]);
        }

        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    let submission_index = queue.submit([encoder.finish()]);

    // Check for errors
    if let Some(error) = futures::executor::block_on(device.pop_error_scope()) {
        return Err(format!("Compute dispatch error: {:?}", error));
    }

    Ok(submission_index)
}
```

## Best Practices

1. **Calculate workgroups carefully** - Ensure all data is processed
2. **Use appropriate workgroup sizes** - Match shader @workgroup_size
3. **Minimize compute passes** - Combine operations when possible
4. **Profile with timestamps** - Identify performance bottlenecks
5. **Handle memory dependencies** - Use separate passes for dependent operations
6. **Use labels consistently** - Essential for debugging
7. **Validate dispatch parameters** - Check against device limits
8. **Consider indirect dispatch** - For dynamic workload sizes

## Common Pitfalls

1. **Forgetting ceiling division** - Use `(n + size - 1) / size`
2. **Exceeding device limits** - Always check max workgroup dimensions
3. **Memory hazards** - Avoid read/write conflicts in same pass
4. **Synchronization issues** - Use proper barriers between dependent operations
5. **Resource binding errors** - Ensure bind groups match pipeline layout