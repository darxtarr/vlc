# ComputePipeline Creation and Management - WGPU 26.0

## Overview

ComputePipeline represents a compute shader with its configuration. It defines how compute work is dispatched and executed on the GPU.

## Basic Pipeline Creation

```rust
use wgpu::{
    Device, ShaderModule, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayout
};

pub fn create_compute_pipeline(
    device: &Device,
    shader_source: &str,
    entry_point: &str,
    label: Option<&str>,
) -> ComputePipeline {
    // 1. Create shader module
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{} Shader", label.unwrap_or("Compute"))),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // 2. Create compute pipeline
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label,
        layout: None, // Use automatic layout generation
        module: &shader_module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None, // Optional pipeline cache for faster compilation
    })
}
```

## ComputePipelineDescriptor Fields

```rust
pub struct ComputePipelineDescriptor<'a> {
    pub label: Label<'a>,                              // Debug label
    pub layout: Option<&'a PipelineLayout>,           // Custom layout or None for auto
    pub module: &'a ShaderModule,                     // Compiled shader
    pub entry_point: Option<&'a str>,                 // Entry function name
    pub compilation_options: PipelineCompilationOptions<'a>, // Advanced options
    pub cache: Option<&'a PipelineCache>,             // Pipeline cache
}
```

### Layout Options

**Automatic Layout (Recommended):**
```rust
let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Auto Layout Pipeline"),
    layout: None, // WGPU generates layout from shader
    module: &shader_module,
    entry_point: Some("main"),
    compilation_options: Default::default(),
    cache: None,
});

// Access generated bind group layouts
let bind_group_layout_0 = pipeline.get_bind_group_layout(0);
let bind_group_layout_1 = pipeline.get_bind_group_layout(1);
```

**Custom Layout:**
```rust
use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, ShaderStages};

// Create custom bind group layout
let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
    label: Some("Custom Compute Layout"),
    entries: &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZero::new(16),
            },
            count: None,
        },
    ],
});

// Create pipeline layout
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("Custom Pipeline Layout"),
    bind_group_layouts: &[&bind_group_layout],
    push_constant_ranges: &[],
});

// Use custom layout in pipeline
let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Custom Layout Pipeline"),
    layout: Some(&pipeline_layout),
    module: &shader_module,
    entry_point: Some("main"),
    compilation_options: Default::default(),
    cache: None,
});
```

## Shader Module Creation

```rust
use wgpu::{ShaderModule, ShaderModuleDescriptor, ShaderSource};

// From WGSL string
fn create_shader_from_wgsl(device: &Device, source: &str, label: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    })
}

// From file
fn create_shader_from_file(device: &Device, path: &str, label: &str) -> ShaderModule {
    let source = std::fs::read_to_string(path)
        .expect("Failed to read shader file");
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    })
}

// Using include_str! macro (compile-time)
fn create_embedded_shader(device: &Device, label: &str) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
    })
}
```

## Pipeline Compilation Options

```rust
use wgpu::{PipelineCompilationOptions, ShaderDefineValue};

let compilation_options = PipelineCompilationOptions {
    constants: [
        ("WORKGROUP_SIZE".to_string(), ShaderDefineValue::U32(256)),
        ("USE_OPTIMIZATIONS".to_string(), ShaderDefineValue::Bool(true)),
    ].iter().cloned().collect(),
    zero_initialize_workgroup_memory: true, // Initialize workgroup memory to zero
    vertex_pulling_transform: false, // Not relevant for compute
};

let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Optimized Pipeline"),
    layout: None,
    module: &shader_module,
    entry_point: Some("main"),
    compilation_options,
    cache: None,
});
```

## Pipeline Cache (Performance Optimization)

```rust
use wgpu::PipelineCache;

// Create pipeline cache for faster subsequent compilations
let pipeline_cache = device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
    label: Some("Compute Pipeline Cache"),
    data: None, // Can load from previously saved cache data
    fallback: true, // Allow fallback if cache is invalid
});

// Use cache in pipeline creation
let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Cached Pipeline"),
    layout: None,
    module: &shader_module,
    entry_point: Some("main"),
    compilation_options: Default::default(),
    cache: Some(&pipeline_cache),
});

// Save cache data for next session
let cache_data = pipeline_cache.get_data();
// Save cache_data to file for reuse
```

## Example WGSL Compute Shader

```wgsl
// compute.wgsl
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    length: u32,
    scale: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.length {
        return;
    }

    data[index] *= params.scale;
}
```

## Best Practices

1. **Use automatic layout** when possible for simpler code
2. **Cache pipelines** for frequently used shaders
3. **Use pipeline cache** to reduce compilation time
4. **Validate entry points** exist in shader modules
5. **Use descriptive labels** for debugging

## Error Handling

```rust
use wgpu::{CreateShaderModuleError, CreatePipelineError};

pub fn safe_create_pipeline(
    device: &Device,
    shader_source: &str,
    entry_point: &str,
) -> Result<ComputePipeline, Box<dyn std::error::Error>> {
    // Validate shader compilation
    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Validated Shader"),
        source: ShaderSource::Wgsl(shader_source.into()),
    });

    // Create pipeline with error handling
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Safe Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    Ok(pipeline)
}
```

## Multiple Entry Points

```rust
// Shader with multiple entry points
let multi_entry_shader = r#"
@compute @workgroup_size(256)
fn process_step1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // First processing step
}

@compute @workgroup_size(128)
fn process_step2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Second processing step
}
"#;

// Create multiple pipelines from same shader
let pipeline1 = create_compute_pipeline(device, multi_entry_shader, "process_step1", Some("Step1"));
let pipeline2 = create_compute_pipeline(device, multi_entry_shader, "process_step2", Some("Step2"));
```