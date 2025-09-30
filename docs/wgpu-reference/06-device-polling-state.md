# Device Polling and State Maintenance - WGPU 26.0

## Overview

Device polling is essential for WGPU applications to handle GPU operations, buffer mapping callbacks, error handling, and resource cleanup. Proper state maintenance ensures optimal performance and prevents resource leaks.

## Device Polling Methods

### Basic Polling

```rust
use wgpu::{Device, Maintain};

// Poll once and return immediately
fn poll_once(device: &Device) {
    device.poll(Maintain::Poll);
}

// Poll and wait for all submitted work to complete
fn poll_and_wait(device: &Device) {
    device.poll(Maintain::Wait);
}

// Poll and wait for specific submission to complete
fn poll_wait_for_submission(device: &Device, submission_index: wgpu::SubmissionIndex) {
    device.poll(Maintain::WaitForSubmissionIndex(submission_index));
}
```

### Maintain Modes

```rust
use wgpu::Maintain;

// Different polling strategies
pub enum PollingStrategy {
    /// Poll once and return immediately
    NonBlocking,
    /// Poll until all submitted work is done
    BlockUntilComplete,
    /// Poll until specific submission is done
    BlockForSubmission(wgpu::SubmissionIndex),
}

fn poll_with_strategy(device: &Device, strategy: PollingStrategy) {
    match strategy {
        PollingStrategy::NonBlocking => {
            device.poll(Maintain::Poll);
        }
        PollingStrategy::BlockUntilComplete => {
            device.poll(Maintain::Wait);
        }
        PollingStrategy::BlockForSubmission(index) => {
            device.poll(Maintain::WaitForSubmissionIndex(index));
        }
    }
}
```

## Async Operations and Callbacks

### Buffer Mapping Callbacks

```rust
use futures_intrusive::channel::shared::oneshot_channel;

async fn async_buffer_operation(
    device: &Device,
    queue: &Queue,
    source_buffer: &Buffer,
    size: u64,
) -> Result<Vec<u8>, wgpu::BufferAsyncError> {
    // Create staging buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Async Read Staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy data
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, size);
    queue.submit([encoder.finish()]);

    // Setup async mapping
    let (sender, receiver) = oneshot_channel();
    staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // Poll until mapping completes
    device.poll(Maintain::Wait);
    receiver.receive().await.unwrap()?;

    // Read data
    let data = staging_buffer.slice(..).get_mapped_range().to_vec();
    staging_buffer.unmap();

    Ok(data)
}
```

### Queue Work Done Callbacks

```rust
use std::sync::{Arc, Mutex};

// Simple callback for work completion
fn setup_work_done_callback(queue: &Queue) {
    queue.on_submitted_work_done(|| {
        println!("GPU work completed!");
    });
}

// Callback with shared state
fn setup_shared_state_callback(queue: &Queue, counter: Arc<Mutex<u32>>) {
    queue.on_submitted_work_done(move || {
        let mut count = counter.lock().unwrap();
        *count += 1;
        println!("Completed operations: {}", *count);
    });
}

// Async notification
async fn wait_for_gpu_work_completion(queue: &Queue) {
    let (sender, receiver) = oneshot_channel();
    queue.on_submitted_work_done(move || {
        sender.send(()).unwrap();
    });

    // Continue other work...

    // Wait for completion
    receiver.receive().await.unwrap();
    println!("All GPU work is done!");
}
```

## Error Handling and Scopes

### Error Scopes

```rust
use wgpu::{ErrorFilter, Error};

// Basic error scope usage
async fn operation_with_error_handling(device: &Device) -> Result<(), String> {
    // Push error scope
    device.push_error_scope(ErrorFilter::Validation);

    // Perform operations that might error
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Pop and check for errors
    if let Some(error) = device.pop_error_scope().await {
        return Err(format!("Operation failed: {:?}", error));
    }

    Ok(())
}

// Multiple error scopes
async fn nested_error_scopes(device: &Device) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Validation errors
    device.push_error_scope(ErrorFilter::Validation);
    // ... validation-sensitive operations ...
    if let Some(error) = device.pop_error_scope().await {
        errors.push(format!("Validation error: {:?}", error));
    }

    // Out of memory errors
    device.push_error_scope(ErrorFilter::OutOfMemory);
    // ... memory-intensive operations ...
    if let Some(error) = device.pop_error_scope().await {
        errors.push(format!("Out of memory: {:?}", error));
    }

    // Internal errors
    device.push_error_scope(ErrorFilter::Internal);
    // ... operations that might cause internal errors ...
    if let Some(error) = device.pop_error_scope().await {
        errors.push(format!("Internal error: {:?}", error));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
```

### Uncaptured Error Handling

```rust
use std::sync::Arc;

fn setup_global_error_handler(device: &Device) {
    device.on_uncaptured_error(Box::new(|error| {
        eprintln!("Uncaptured GPU error: {:?}", error);

        match error {
            Error::OutOfMemory { .. } => {
                eprintln!("GPU ran out of memory!");
                // Handle memory exhaustion
            }
            Error::Validation { description, .. } => {
                eprintln!("Validation error: {}", description);
                // Handle validation failures
            }
            Error::Internal { description, .. } => {
                eprintln!("Internal GPU error: {}", description);
                // Handle internal errors
            }
        }
    }));
}
```

### Device Lost Handling

```rust
use wgpu::{DeviceLostReason, DeviceLostCallback};

fn setup_device_lost_callback(device: &Device) {
    device.set_device_lost_callback(Box::new(|reason, message| {
        match reason {
            DeviceLostReason::Unknown => {
                eprintln!("Device lost for unknown reason: {}", message);
            }
            DeviceLostReason::Destroyed => {
                eprintln!("Device was explicitly destroyed: {}", message);
            }
        }

        // Handle device loss - usually requires recreation
        // trigger_device_recreation();
    }));
}
```

## Resource Management

### Automatic Resource Cleanup

```rust
pub struct ResourceManager {
    device: Arc<Device>,
    buffers: Vec<Buffer>,
    textures: Vec<wgpu::Texture>,
    bind_groups: Vec<BindGroup>,
}

impl ResourceManager {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            buffers: Vec::new(),
            textures: Vec::new(),
            bind_groups: Vec::new(),
        }
    }

    pub fn create_buffer(&mut self, descriptor: &wgpu::BufferDescriptor) -> &Buffer {
        let buffer = self.device.create_buffer(descriptor);
        self.buffers.push(buffer);
        self.buffers.last().unwrap()
    }

    pub fn cleanup(&mut self) {
        // Explicit cleanup if needed
        for buffer in &self.buffers {
            buffer.destroy();
        }
        for texture in &self.textures {
            texture.destroy();
        }

        self.buffers.clear();
        self.textures.clear();
        self.bind_groups.clear();

        // Poll to ensure cleanup is processed
        self.device.poll(Maintain::Wait);
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        self.cleanup();
    }
}
```

### Memory Usage Monitoring

```rust
use std::time::{Duration, Instant};

pub struct MemoryMonitor {
    device: Arc<Device>,
    last_report_time: Instant,
    report_interval: Duration,
}

impl MemoryMonitor {
    pub fn new(device: Arc<Device>, report_interval_secs: u64) -> Self {
        Self {
            device,
            last_report_time: Instant::now(),
            report_interval: Duration::from_secs(report_interval_secs),
        }
    }

    pub fn poll_and_maybe_report(&mut self) {
        self.device.poll(Maintain::Poll);

        if self.last_report_time.elapsed() >= self.report_interval {
            self.generate_memory_report();
            self.last_report_time = Instant::now();
        }
    }

    fn generate_memory_report(&self) {
        if let Some(report) = self.device.generate_allocator_report() {
            println!("GPU Memory Report:");
            println!("  Total allocated: {} bytes", report.total_allocated_bytes);
            println!("  Total deallocated: {} bytes", report.total_deallocated_bytes);
            println!("  Active allocations: {}", report.allocations.len());

            // Log large allocations
            for allocation in &report.allocations {
                if allocation.size > 1024 * 1024 { // > 1MB
                    println!("  Large allocation: {} bytes at {:?}",
                             allocation.size, allocation.name);
                }
            }
        }
    }
}
```

## Application Main Loop Patterns

### Game Loop with GPU Operations

```rust
use std::time::{Duration, Instant};

pub struct GPUGameLoop {
    device: Arc<Device>,
    queue: Arc<Queue>,
    last_frame_time: Instant,
    target_fps: u32,
}

impl GPUGameLoop {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, target_fps: u32) -> Self {
        Self {
            device,
            queue,
            last_frame_time: Instant::now(),
            target_fps,
        }
    }

    pub fn run<F>(&mut self, mut frame_callback: F)
    where
        F: FnMut(&Device, &Queue, f32) -> Result<(), Box<dyn std::error::Error>>,
    {
        let frame_duration = Duration::from_nanos(1_000_000_000 / self.target_fps as u64);

        loop {
            let frame_start = Instant::now();
            let delta_time = frame_start.duration_since(self.last_frame_time).as_secs_f32();

            // Process GPU operations
            self.device.poll(Maintain::Poll);

            // Execute frame
            if let Err(e) = frame_callback(&self.device, &self.queue, delta_time) {
                eprintln!("Frame error: {}", e);
                break;
            }

            // Frame rate limiting
            let frame_time = frame_start.elapsed();
            if frame_time < frame_duration {
                std::thread::sleep(frame_duration - frame_time);
            }

            self.last_frame_time = frame_start;
        }
    }
}
```

### Async Compute Pipeline

```rust
use tokio::time::{sleep, Duration};

pub struct AsyncComputeManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl AsyncComputeManager {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub async fn start_background_processing(&self) {
        self.running.store(true, std::sync::atomic::Ordering::Relaxed);

        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            while running.load(std::sync::atomic::Ordering::Relaxed) {
                // Poll device for completed operations
                device.poll(Maintain::Poll);

                // Small delay to prevent busy waiting
                sleep(Duration::from_millis(1)).await;
            }
        });
    }

    pub fn stop(&self) {
        self.running.store(false, std::sync::atomic::Ordering::Relaxed);

        // Final poll to clean up
        self.device.poll(Maintain::Wait);
    }

    pub async fn submit_compute_work<F>(&self, work: F) -> Result<(), String>
    where
        F: FnOnce(&Device, &Queue) -> Result<wgpu::SubmissionIndex, String>,
    {
        let submission_index = work(&self.device, &self.queue)?;

        // Wait for this specific work to complete
        self.device.poll(Maintain::WaitForSubmissionIndex(submission_index));

        Ok(())
    }
}
```

## Performance Optimization

### Batched Polling

```rust
pub struct BatchedPoller {
    device: Arc<Device>,
    poll_counter: u32,
    poll_threshold: u32,
}

impl BatchedPoller {
    pub fn new(device: Arc<Device>, poll_threshold: u32) -> Self {
        Self {
            device,
            poll_counter: 0,
            poll_threshold,
        }
    }

    pub fn maybe_poll(&mut self) {
        self.poll_counter += 1;

        if self.poll_counter >= self.poll_threshold {
            self.device.poll(Maintain::Poll);
            self.poll_counter = 0;
        }
    }

    pub fn force_poll(&mut self) {
        self.device.poll(Maintain::Poll);
        self.poll_counter = 0;
    }

    pub fn wait_for_completion(&mut self) {
        self.device.poll(Maintain::Wait);
        self.poll_counter = 0;
    }
}
```

### Adaptive Polling

```rust
use std::time::{Duration, Instant};

pub struct AdaptivePoller {
    device: Arc<Device>,
    last_poll_time: Instant,
    poll_interval: Duration,
    min_interval: Duration,
    max_interval: Duration,
    pending_operations: u32,
}

impl AdaptivePoller {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            last_poll_time: Instant::now(),
            poll_interval: Duration::from_millis(1),
            min_interval: Duration::from_micros(100),
            max_interval: Duration::from_millis(16),
            pending_operations: 0,
        }
    }

    pub fn add_pending_operation(&mut self) {
        self.pending_operations += 1;
        // Decrease interval when more work is pending
        self.poll_interval = self.min_interval;
    }

    pub fn maybe_poll(&mut self) {
        if self.last_poll_time.elapsed() >= self.poll_interval {
            self.device.poll(Maintain::Poll);
            self.last_poll_time = Instant::now();

            // Adaptive interval adjustment
            if self.pending_operations > 0 {
                self.pending_operations = self.pending_operations.saturating_sub(1);
            }

            if self.pending_operations == 0 {
                // Increase interval when no pending work
                self.poll_interval = (self.poll_interval * 2).min(self.max_interval);
            }
        }
    }
}
```

## Best Practices

1. **Poll regularly** - Don't let the GPU queue back up
2. **Use appropriate Maintain modes** - Poll for regular updates, Wait for synchronization
3. **Handle device loss** - Set up callbacks for robust applications
4. **Monitor memory usage** - Use allocator reports for debugging
5. **Use error scopes** - Catch validation and other errors early
6. **Batch operations** - Reduce polling overhead
7. **Async patterns** - Don't block the main thread unnecessarily
8. **Resource cleanup** - Explicitly destroy resources when done

## Common Pitfalls

1. **Not polling enough** - Callbacks won't fire, memory won't be freed
2. **Polling too much** - Unnecessary CPU overhead
3. **Ignoring device loss** - Application crashes instead of graceful handling
4. **Forgetting error scopes** - Silent failures in validation
5. **Resource leaks** - Not destroying buffers/textures when done
6. **Blocking main thread** - Using Maintain::Wait in UI applications