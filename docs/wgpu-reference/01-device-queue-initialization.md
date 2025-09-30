# Device and Queue Initialization - WGPU 26.0

## Overview

Device and Queue initialization is the foundation of any WGPU application. The pattern involves creating an Instance, requesting an Adapter, and then creating a Device and Queue.

## Basic Initialization Pattern

```rust
use wgpu::{
    Instance, RequestAdapterOptions, DeviceDescriptor, Features,
    Limits, PowerPreference, Device, Queue
};

pub async fn initialize_gpu() -> Result<(Device, Queue), Box<dyn std::error::Error>> {
    // 1. Create Instance - entry point to WGPU
    let instance = Instance::default();

    // 2. Request Adapter - represents physical GPU
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None, // None for compute-only applications
            force_fallback_adapter: false,
        })
        .await
        .ok_or("Failed to find suitable adapter")?;

    // 3. Create Device and Queue
    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("Compute Device"),
                required_features: Features::empty(), // Add compute features as needed
                required_limits: Limits::default(),
                memory_hints: Default::default(), // New in WGPU 26.0
                trace: wgpu::Trace::None, // For debugging/profiling
            },
            None, // trace_path for debugging
        )
        .await?;

    Ok((device, queue))
}
```

## Key Components

### Instance

The `Instance` is the entry point to WGPU. Use `Instance::default()` for most cases.

```rust
let instance = Instance::default();
```

### RequestAdapterOptions

Configure adapter selection:

```rust
let adapter_options = RequestAdapterOptions {
    power_preference: PowerPreference::HighPerformance, // or LowPower
    compatible_surface: None, // Required for rendering, None for compute
    force_fallback_adapter: false, // Force software fallback
};
```

### DeviceDescriptor

Configure device capabilities:

```rust
let device_descriptor = DeviceDescriptor {
    label: Some("My Compute Device"),
    required_features: Features::empty(), // See features section below
    required_limits: Limits::default(), // Or custom limits
    memory_hints: Default::default(), // Memory allocation hints
    trace: wgpu::Trace::None, // Debugging/profiling trace
};
```

## Important Features for Compute

```rust
use wgpu::Features;

// Common compute features
let features = Features::TIMESTAMP_QUERY // For performance profiling
    | Features::PIPELINE_STATISTICS_QUERY // For pipeline statistics
    | Features::BUFFER_BINDING_ARRAY // For buffer arrays in shaders
    | Features::TEXTURE_BINDING_ARRAY; // For texture arrays in shaders
```

## Error Handling

```rust
pub async fn robust_gpu_init() -> Result<(Device, Queue), String> {
    let instance = Instance::default();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No suitable GPU adapter found")?;

    // Check adapter capabilities
    let limits = adapter.limits();
    println!("Max compute workgroups per dimension: {}", limits.max_compute_workgroups_per_dimension);
    println!("Max compute workgroup size X: {}", limits.max_compute_workgroup_size_x);
    println!("Max storage buffer binding size: {}", limits.max_storage_buffer_binding_size);

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("VLC Compute Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::None,
            },
            None,
        )
        .await
        .map_err(|e| format!("Failed to create device: {}", e))?;

    Ok((device, queue))
}
```

## Best Practices

1. **Use Arc for sharing**: Wrap Device and Queue in `Arc` for multi-threaded access
   ```rust
   use std::sync::Arc;
   let device = Arc::new(device);
   let queue = Arc::new(queue);
   ```

2. **Check adapter limits**: Always verify the adapter supports your requirements
   ```rust
   let limits = adapter.limits();
   if limits.max_storage_buffer_binding_size < required_buffer_size {
       return Err("Insufficient buffer size support");
   }
   ```

3. **Handle device loss**: Set up device lost callback for robust applications
   ```rust
   device.set_device_lost_callback(Box::new(|reason, message| {
       eprintln!("Device lost: {:?} - {}", reason, message);
   }));
   ```

4. **Memory hints** (New in 26.0): Use memory hints for optimization
   ```rust
   let memory_hints = wgpu::MemoryHints {
       usage: wgpu::MemoryUsage::GpuOnly, // or CpuToGpu, GpuToCpu
   };
   ```

## Changes from Previous Versions

- **memory_hints**: New field in DeviceDescriptor for memory allocation optimization
- **trace**: Now uses `wgpu::Trace` enum instead of optional path
- **Features**: Some features have been added/renamed for ray tracing and advanced compute