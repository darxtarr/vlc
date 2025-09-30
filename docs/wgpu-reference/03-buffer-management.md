# Buffer Creation, Management, and Data Transfer - WGPU 26.0

## Overview

Buffers are the primary way to store and transfer data between CPU and GPU. WGPU 26.0 provides comprehensive buffer management with various usage patterns and efficient data transfer mechanisms.

## Buffer Creation

### Basic Buffer Creation

```rust
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

fn create_buffer(device: &Device, size: u64, usage: BufferUsages, label: Option<&str>) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label,
        size,
        usage,
        mapped_at_creation: false,
    })
}
```

### BufferUsages Patterns

```rust
use wgpu::BufferUsages;

// Storage buffer for compute shaders (read/write)
let storage_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Storage Buffer"),
    size: 1024 * 1024, // 1MB
    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Uniform buffer for shader parameters
let uniform_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Uniform Buffer"),
    size: 256,
    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Staging buffer for CPU-GPU data transfer
let staging_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Staging Buffer"),
    size: 1024 * 1024,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Vertex buffer (if needed for graphics)
let vertex_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Vertex Buffer"),
    size: 4096,
    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Index buffer (if needed for graphics)
let index_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Index Buffer"),
    size: 2048,
    usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

### Buffer Creation with Initial Data

```rust
// Create buffer with data at creation time
fn create_buffer_with_data<T: bytemuck::Pod>(
    device: &Device,
    data: &[T],
    usage: BufferUsages,
    label: Option<&str>,
) -> Buffer {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let buffer = device.create_buffer(&BufferDescriptor {
        label,
        size,
        usage,
        mapped_at_creation: true,
    });

    // Write data during creation
    buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(data));
    buffer.unmap();

    buffer
}

// Example usage
let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let buffer = create_buffer_with_data(
    device,
    &data,
    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    Some("Initial Data Buffer"),
);
```

## Data Transfer Methods

### 1. Queue Write Buffer (Recommended for frequent updates)

```rust
use wgpu::Queue;

// Direct write to buffer (most efficient for frequent updates)
fn write_buffer_data<T: bytemuck::Pod>(
    queue: &Queue,
    buffer: &Buffer,
    offset: u64,
    data: &[T],
) {
    queue.write_buffer(buffer, offset, bytemuck::cast_slice(data));
}

// Example
let uniform_data = [1.0f32, 2.0, 3.0, 4.0];
queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));
```

### 2. Queue Write Buffer With (For large data)

```rust
// For large buffers where you want to write directly to staging memory
fn write_large_buffer_data<T: bytemuck::Pod>(
    queue: &Queue,
    buffer: &Buffer,
    offset: u64,
    data: &[T],
) -> Option<()> {
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    let mut write_view = queue.write_buffer_with(buffer, offset, size.try_into().ok()?)?;

    // Write directly to the staging buffer
    write_view.copy_from_slice(bytemuck::cast_slice(data));

    Some(())
}
```

### 3. Buffer Mapping (For reading data back)

```rust
use futures_intrusive::channel::shared::oneshot_channel;

// Async buffer mapping for reading data
async fn read_buffer_data<T: bytemuck::Pod + Clone>(
    device: &Device,
    queue: &Queue,
    source_buffer: &Buffer,
    size: u64,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    // Create staging buffer for reading
    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Read Staging Buffer"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy from source to staging
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, size);
    queue.submit([encoder.finish()]);

    // Map staging buffer for reading
    let (sender, receiver) = oneshot_channel();
    staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // Poll device and wait for mapping
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap()?;

    // Read data
    let data_slice = staging_buffer.slice(..).get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data_slice).to_vec();

    // Cleanup
    drop(data_slice);
    staging_buffer.unmap();

    Ok(result)
}
```

### 4. Buffer-to-Buffer Copies

```rust
use wgpu::CommandEncoder;

fn copy_buffer_data(
    device: &Device,
    queue: &Queue,
    source: &Buffer,
    dest: &Buffer,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Buffer Copy Encoder"),
    });

    encoder.copy_buffer_to_buffer(source, src_offset, dest, dst_offset, size);
    queue.submit([encoder.finish()]);
}
```

## Buffer Slicing

```rust
use wgpu::{BufferSlice, BufferAddress};

// Get slice of entire buffer
let full_slice = buffer.slice(..);

// Get slice with specific range
let partial_slice = buffer.slice(64..128);

// Get slice with BufferAddress types
let typed_slice = buffer.slice(start_offset..end_offset);

// Using slices for mapping
async fn map_buffer_slice(slice: BufferSlice<'_>) -> Result<(), wgpu::BufferAsyncError> {
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    receiver.receive().await.unwrap()
}
```

## Advanced Buffer Patterns

### Buffer Pool for Reuse

```rust
use std::collections::VecDeque;

pub struct BufferPool {
    device: std::sync::Arc<Device>,
    available_buffers: VecDeque<Buffer>,
    buffer_size: u64,
    usage: BufferUsages,
}

impl BufferPool {
    pub fn new(
        device: std::sync::Arc<Device>,
        initial_count: usize,
        buffer_size: u64,
        usage: BufferUsages,
    ) -> Self {
        let mut available_buffers = VecDeque::new();

        for i in 0..initial_count {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("Pool Buffer {}", i)),
                size: buffer_size,
                usage,
                mapped_at_creation: false,
            });
            available_buffers.push_back(buffer);
        }

        Self {
            device,
            available_buffers,
            buffer_size,
            usage,
        }
    }

    pub fn get_buffer(&mut self) -> Buffer {
        self.available_buffers.pop_front().unwrap_or_else(|| {
            // Create new buffer if pool is empty
            self.device.create_buffer(&BufferDescriptor {
                label: Some("Pool Buffer (Dynamic)"),
                size: self.buffer_size,
                usage: self.usage,
                mapped_at_creation: false,
            })
        })
    }

    pub fn return_buffer(&mut self, buffer: Buffer) {
        self.available_buffers.push_back(buffer);
    }
}
```

### Ring Buffer for Streaming Data

```rust
pub struct RingBuffer {
    buffer: Buffer,
    size: u64,
    write_offset: u64,
    frame_size: u64,
}

impl RingBuffer {
    pub fn new(device: &Device, total_size: u64, frame_size: u64) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Ring Buffer"),
            size: total_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size: total_size,
            write_offset: 0,
            frame_size,
        }
    }

    pub fn write_frame<T: bytemuck::Pod>(&mut self, queue: &Queue, data: &[T]) {
        let data_size = (data.len() * std::mem::size_of::<T>()) as u64;
        assert!(data_size <= self.frame_size);

        // Write to current offset
        queue.write_buffer(&self.buffer, self.write_offset, bytemuck::cast_slice(data));

        // Advance offset with wrapping
        self.write_offset = (self.write_offset + self.frame_size) % self.size;
    }

    pub fn get_current_offset(&self) -> u64 {
        self.write_offset
    }
}
```

## Buffer Synchronization

### Explicit Synchronization

```rust
// Ensure all operations complete before proceeding
fn synchronize_buffer_operations(device: &Device, queue: &Queue) {
    // Submit any pending operations
    queue.submit(std::iter::empty());

    // Poll until complete
    device.poll(wgpu::Maintain::Wait);
}

// Callback-based synchronization
fn async_buffer_operation_complete(queue: &Queue, callback: impl FnOnce() + Send + 'static) {
    queue.on_submitted_work_done(callback);
}
```

## Error Handling

```rust
use wgpu::{BufferAsyncError, MapMode};

async fn safe_buffer_read<T: bytemuck::Pod + Clone>(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    offset: u64,
    size: u64,
) -> Result<Vec<T>, String> {
    // Validate buffer size
    if offset + size > buffer.size() {
        return Err("Buffer read out of bounds".to_string());
    }

    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Safe Read Staging"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy with error handling
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, offset, &staging_buffer, 0, size);
    queue.submit([encoder.finish()]);

    // Map with timeout handling
    let (sender, receiver) = oneshot_channel();
    staging_buffer.slice(..).map_async(MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    device.poll(wgpu::Maintain::Wait);

    match receiver.receive().await.unwrap() {
        Ok(()) => {
            let data_slice = staging_buffer.slice(..).get_mapped_range();
            let result: Vec<T> = bytemuck::cast_slice(&data_slice).to_vec();
            drop(data_slice);
            staging_buffer.unmap();
            Ok(result)
        }
        Err(BufferAsyncError::DestroyedResource) => {
            Err("Buffer was destroyed during mapping".to_string())
        }
        Err(BufferAsyncError::DeviceLost) => {
            Err("Device lost during buffer mapping".to_string())
        }
        Err(BufferAsyncError::Unknown) => {
            Err("Unknown error during buffer mapping".to_string())
        }
        Err(BufferAsyncError::ValidationError(msg)) => {
            Err(format!("Validation error: {}", msg))
        }
    }
}
```

## Best Practices

1. **Use appropriate BufferUsages** - Only include necessary usage flags
2. **Prefer queue.write_buffer()** for frequent CPU->GPU transfers
3. **Use staging buffers** for GPU->CPU transfers
4. **Buffer pooling** for frequently allocated/deallocated buffers
5. **Align buffer sizes** to GPU requirements (typically 4-byte aligned)
6. **Cleanup mapped buffers** by calling unmap() when done
7. **Use labels** for debugging buffer usage
8. **Validate buffer bounds** before operations

## Performance Tips

1. **Batch transfers** - Combine multiple small writes into fewer large ones
2. **Minimize CPU-GPU sync points** - Use async operations when possible
3. **Reuse buffers** instead of frequent allocation/deallocation
4. **Use persistent mapping** for frequently updated uniform buffers
5. **Align data** to GPU cache line boundaries (64-256 bytes)