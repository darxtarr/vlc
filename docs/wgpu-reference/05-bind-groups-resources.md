# Bind Groups and Resource Binding - WGPU 26.0

## Overview

Bind groups are collections of resources (buffers, textures, samplers) that are bound together and made available to shaders. They provide an efficient way to manage and bind multiple resources to shader stages.

## Basic Bind Group Creation

```rust
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource,
    BufferBinding, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, ShaderStages, Device, Buffer
};

fn create_simple_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    storage_buffer: &Buffer,
    uniform_buffer: &Buffer,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some("Simple Bind Group"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: storage_buffer,
                    offset: 0,
                    size: None, // Use entire buffer
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: std::num::NonZero::new(256), // Specific size
                }),
            },
        ],
    })
}
```

## Bind Group Layout Creation

### Manual Layout Creation

```rust
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ShaderStages, Features
};

fn create_compute_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            // Storage buffer (read-write)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniform buffer (read-only)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZero::new(16),
                },
                count: None,
            },
            // Read-only storage buffer
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}
```

### Automatic Layout from Pipeline

```rust
fn get_layout_from_pipeline(pipeline: &ComputePipeline, group_index: u32) -> BindGroupLayout {
    pipeline.get_bind_group_layout(group_index)
}

// Example usage
let layout = get_layout_from_pipeline(&compute_pipeline, 0);
let bind_group = create_bind_group_from_layout(device, &layout, buffers);
```

## Resource Binding Types

### Buffer Bindings

```rust
// Storage buffer binding (read-write)
let storage_binding = BindingResource::Buffer(BufferBinding {
    buffer: &storage_buffer,
    offset: 0,
    size: None, // Use entire buffer
});

// Uniform buffer binding with specific size
let uniform_binding = BindingResource::Buffer(BufferBinding {
    buffer: &uniform_buffer,
    offset: 64, // Start at offset 64
    size: std::num::NonZero::new(256), // Use 256 bytes
});

// Dynamic offset binding (offset specified at bind time)
let dynamic_binding = BindingResource::Buffer(BufferBinding {
    buffer: &buffer,
    offset: 0, // Base offset
    size: std::num::NonZero::new(1024),
});
```

### Buffer Array Bindings

```rust
use wgpu::Features;

// Requires BUFFER_BINDING_ARRAY feature
fn create_buffer_array_binding(buffers: &[&Buffer]) -> BindingResource {
    let buffer_bindings: Vec<BufferBinding> = buffers
        .iter()
        .map(|buffer| BufferBinding {
            buffer,
            offset: 0,
            size: None,
        })
        .collect();

    BindingResource::BufferArray(&buffer_bindings)
}

// Layout for buffer array
let buffer_array_layout = BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: std::num::NonZero::new(4), // Array of 4 buffers
};
```

### Texture and Sampler Bindings

```rust
use wgpu::{TextureView, Sampler, TextureViewDimension, TextureSampleType};

// Texture binding
let texture_binding = BindingResource::TextureView(&texture_view);

// Sampler binding
let sampler_binding = BindingResource::Sampler(&sampler);

// Texture array binding (requires TEXTURE_BINDING_ARRAY feature)
let texture_array_binding = BindingResource::TextureViewArray(&[&texture_view1, &texture_view2]);

// Sampler array binding
let sampler_array_binding = BindingResource::SamplerArray(&[&sampler1, &sampler2]);

// Corresponding layout entries
let texture_layout = BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Texture {
        sample_type: TextureSampleType::Float { filterable: true },
        view_dimension: TextureViewDimension::D2,
        multisampled: false,
    },
    count: None,
};

let sampler_layout = BindGroupLayoutEntry {
    binding: 1,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    count: None,
};
```

## Multiple Bind Groups

```rust
fn create_multi_group_setup(
    device: &Device,
    pipelines: &ComputePipeline,
) -> (Vec<BindGroupLayout>, Vec<BindGroup>) {
    // Get layouts from pipeline
    let layout_0 = pipelines.get_bind_group_layout(0);
    let layout_1 = pipelines.get_bind_group_layout(1);

    // Create bind groups
    let bind_group_0 = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group 0 - Data"),
        layout: &layout_0,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &input_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &output_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    let bind_group_1 = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Bind Group 1 - Parameters"),
        layout: &layout_1,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: std::num::NonZero::new(64),
                }),
            },
        ],
    });

    (vec![layout_0, layout_1], vec![bind_group_0, bind_group_1])
}

// Usage in compute pass
fn use_multiple_bind_groups(
    compute_pass: &mut ComputePass,
    bind_groups: &[BindGroup],
) {
    for (index, bind_group) in bind_groups.iter().enumerate() {
        compute_pass.set_bind_group(index as u32, bind_group, &[]);
    }
}
```

## Dynamic Offsets

```rust
// Layout with dynamic offset
let dynamic_layout = BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: true, // Enable dynamic offset
        min_binding_size: std::num::NonZero::new(256),
    },
    count: None,
};

// Bind group with dynamic offset
let bind_group = device.create_bind_group(&BindGroupDescriptor {
    label: Some("Dynamic Offset Bind Group"),
    layout: &layout,
    entries: &[BindGroupEntry {
        binding: 0,
        resource: BindingResource::Buffer(BufferBinding {
            buffer: &uniform_buffer,
            offset: 0, // Base offset
            size: std::num::NonZero::new(256),
        }),
    }],
});

// Set bind group with dynamic offset
fn set_bind_group_with_offset(
    compute_pass: &mut ComputePass,
    bind_group: &BindGroup,
    dynamic_offsets: &[u32],
) {
    compute_pass.set_bind_group(0, bind_group, dynamic_offsets);
}

// Example: Different offsets for different frames
let frame_offsets = [0, 256, 512, 768]; // Different uniform data per frame
compute_pass.set_bind_group(0, &bind_group, &[frame_offsets[frame_index]]);
```

## Bind Group Builder Pattern

```rust
pub struct BindGroupBuilder<'a> {
    device: &'a Device,
    layout: &'a BindGroupLayout,
    entries: Vec<BindGroupEntry<'a>>,
    label: Option<&'a str>,
}

impl<'a> BindGroupBuilder<'a> {
    pub fn new(device: &'a Device, layout: &'a BindGroupLayout) -> Self {
        Self {
            device,
            layout,
            entries: Vec::new(),
            label: None,
        }
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn buffer(mut self, binding: u32, buffer: &'a Buffer) -> Self {
        self.entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::Buffer(BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
        });
        self
    }

    pub fn buffer_with_range(
        mut self,
        binding: u32,
        buffer: &'a Buffer,
        offset: u64,
        size: Option<std::num::NonZero<u64>>,
    ) -> Self {
        self.entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::Buffer(BufferBinding {
                buffer,
                offset,
                size,
            }),
        });
        self
    }

    pub fn texture(mut self, binding: u32, texture_view: &'a TextureView) -> Self {
        self.entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::TextureView(texture_view),
        });
        self
    }

    pub fn sampler(mut self, binding: u32, sampler: &'a Sampler) -> Self {
        self.entries.push(BindGroupEntry {
            binding,
            resource: BindingResource::Sampler(sampler),
        });
        self
    }

    pub fn build(self) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: self.label,
            layout: self.layout,
            entries: &self.entries,
        })
    }
}

// Usage
let bind_group = BindGroupBuilder::new(device, &layout)
    .label("My Bind Group")
    .buffer(0, &storage_buffer)
    .buffer_with_range(1, &uniform_buffer, 0, std::num::NonZero::new(256))
    .texture(2, &texture_view)
    .sampler(3, &sampler)
    .build();
```

## Bind Group Caching

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct BindGroupKey {
    layout_id: u64,
    buffer_ids: Vec<u64>,
    offsets: Vec<u64>,
    sizes: Vec<Option<u64>>,
}

impl Hash for BindGroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layout_id.hash(state);
        self.buffer_ids.hash(state);
        self.offsets.hash(state);
        self.sizes.hash(state);
    }
}

impl PartialEq for BindGroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.layout_id == other.layout_id
            && self.buffer_ids == other.buffer_ids
            && self.offsets == other.offsets
            && self.sizes == other.sizes
    }
}

impl Eq for BindGroupKey {}

pub struct BindGroupCache {
    cache: HashMap<BindGroupKey, BindGroup>,
}

impl BindGroupCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get_or_create(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
        buffers: &[&Buffer],
        offsets: &[u64],
        sizes: &[Option<std::num::NonZero<u64>>],
    ) -> &BindGroup {
        // Create cache key (simplified - would need proper ID extraction)
        let key = BindGroupKey {
            layout_id: layout as *const _ as u64,
            buffer_ids: buffers.iter().map(|b| *b as *const _ as u64).collect(),
            offsets: offsets.to_vec(),
            sizes: sizes.iter().map(|s| s.map(|v| v.get())).collect(),
        };

        self.cache.entry(key).or_insert_with(|| {
            let entries: Vec<BindGroupEntry> = buffers
                .iter()
                .zip(offsets.iter())
                .zip(sizes.iter())
                .enumerate()
                .map(|(i, ((buffer, &offset), size))| BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer,
                        offset,
                        size: *size,
                    }),
                })
                .collect();

            device.create_bind_group(&BindGroupDescriptor {
                label: Some("Cached Bind Group"),
                layout,
                entries: &entries,
            })
        })
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
```

## Error Handling

```rust
fn validate_bind_group_creation(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[BindGroupEntry],
) -> Result<BindGroup, String> {
    // Basic validation
    if entries.is_empty() {
        return Err("Bind group entries cannot be empty".to_string());
    }

    // Create with error scope
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Validated Bind Group"),
        layout,
        entries,
    });

    // Check for errors
    if let Some(error) = futures::executor::block_on(device.pop_error_scope()) {
        return Err(format!("Bind group creation failed: {:?}", error));
    }

    Ok(bind_group)
}
```

## Best Practices

1. **Use automatic layouts** when possible for simpler code
2. **Cache bind groups** for frequently used resource combinations
3. **Group related resources** in the same bind group
4. **Use dynamic offsets** for frame-varying uniform data
5. **Validate resource compatibility** with layouts before creation
6. **Use descriptive labels** for debugging
7. **Minimize bind group changes** during render/compute passes
8. **Consider binding frequency** when organizing resources

## Common Patterns

### Frame-based Uniform Updates

```rust
struct FrameUniforms {
    data: [f32; 64], // 256 bytes
}

fn create_frame_uniform_setup(device: &Device, frame_count: usize) -> (Buffer, BindGroup) {
    // Create buffer large enough for all frames
    let buffer_size = (frame_count * std::mem::size_of::<FrameUniforms>()) as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Frame Uniforms"),
        size: buffer_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create layout with dynamic offset
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Frame Uniform Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZero::new(std::mem::size_of::<FrameUniforms>() as u64),
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Frame Uniform Bind Group"),
        layout: &layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: std::num::NonZero::new(std::mem::size_of::<FrameUniforms>() as u64),
            }),
        }],
    });

    (buffer, bind_group)
}
```

### Multi-Buffer Operations

```rust
fn create_multi_buffer_bind_group(
    device: &Device,
    input_buffers: &[Buffer],
    output_buffer: &Buffer,
    uniform_buffer: &Buffer,
) -> BindGroup {
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Multi-Buffer Layout"),
        entries: &[
            // Input buffer array
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(input_buffers.len() as u32),
            },
            // Output buffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniform parameters
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZero::new(64),
                },
                count: None,
            },
        ],
    });

    let input_bindings: Vec<BufferBinding> = input_buffers
        .iter()
        .map(|buffer| BufferBinding {
            buffer,
            offset: 0,
            size: None,
        })
        .collect();

    device.create_bind_group(&BindGroupDescriptor {
        label: Some("Multi-Buffer Bind Group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::BufferArray(&input_bindings),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: output_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: uniform_buffer,
                    offset: 0,
                    size: std::num::NonZero::new(64),
                }),
            },
        ],
    })
}
```