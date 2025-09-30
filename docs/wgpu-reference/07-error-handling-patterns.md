# Error Handling Patterns - WGPU 26.0

## Overview

Robust error handling is crucial for GPU applications. WGPU 26.0 provides comprehensive error reporting mechanisms including error scopes, device lost callbacks, and async error handling for buffer operations.

## Error Types

### Core Error Types

```rust
use wgpu::{Error, BufferAsyncError, SurfaceError, CreateSurfaceError, RequestDeviceError};

// Main WGPU errors
pub enum WGPUError {
    Validation(String),
    OutOfMemory,
    Internal(String),
    BufferAsync(BufferAsyncError),
    Surface(SurfaceError),
    DeviceRequest(RequestDeviceError),
    CreateSurface(CreateSurfaceError),
}

impl From<Error> for WGPUError {
    fn from(error: Error) -> Self {
        match error {
            Error::OutOfMemory { .. } => WGPUError::OutOfMemory,
            Error::Validation { description, .. } => WGPUError::Validation(description),
            Error::Internal { description, .. } => WGPUError::Internal(description),
        }
    }
}
```

### Buffer Operation Errors

```rust
use wgpu::BufferAsyncError;

fn handle_buffer_error(error: BufferAsyncError) -> String {
    match error {
        BufferAsyncError::DestroyedResource => {
            "Buffer was destroyed before mapping completed".to_string()
        }
        BufferAsyncError::DeviceLost => {
            "Device was lost during buffer operation".to_string()
        }
        BufferAsyncError::Unknown => {
            "Unknown error occurred during buffer operation".to_string()
        }
        BufferAsyncError::ValidationError(msg) => {
            format!("Buffer validation error: {}", msg)
        }
    }
}
```

## Error Scopes

### Basic Error Scope Usage

```rust
use wgpu::{Device, ErrorFilter, Error};

async fn safe_buffer_creation(device: &Device) -> Result<wgpu::Buffer, String> {
    // Push error scope before potentially failing operation
    device.push_error_scope(ErrorFilter::Validation);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Safe Buffer"),
        size: 1024 * 1024 * 1024, // Large size that might fail
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Pop error scope and check for errors
    match device.pop_error_scope().await {
        Some(Error::OutOfMemory { .. }) => {
            Err("Not enough GPU memory for buffer".to_string())
        }
        Some(Error::Validation { description, .. }) => {
            Err(format!("Buffer validation failed: {}", description))
        }
        Some(Error::Internal { description, .. }) => {
            Err(format!("Internal GPU error: {}", description))
        }
        None => Ok(buffer),
    }
}
```

### Nested Error Scopes

```rust
async fn multi_operation_with_errors(device: &Device) -> Result<Vec<wgpu::Buffer>, Vec<String>> {
    let mut errors = Vec::new();
    let mut buffers = Vec::new();

    // Operation 1: Create large buffer
    device.push_error_scope(ErrorFilter::OutOfMemory);
    let large_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Large Buffer"),
        size: 1024 * 1024 * 512, // 512MB
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    if let Some(error) = device.pop_error_scope().await {
        errors.push(format!("Large buffer creation failed: {:?}", error));
    } else {
        buffers.push(large_buffer);
    }

    // Operation 2: Create buffer with invalid usage
    device.push_error_scope(ErrorFilter::Validation);
    let invalid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::empty(), // Invalid: no usage flags
        mapped_at_creation: false,
    });

    if let Some(error) = device.pop_error_scope().await {
        errors.push(format!("Invalid buffer creation failed: {:?}", error));
    } else {
        buffers.push(invalid_buffer);
    }

    if errors.is_empty() {
        Ok(buffers)
    } else {
        Err(errors)
    }
}
```

### Error Scope Helper

```rust
use std::future::Future;

pub struct ErrorScope<'a> {
    device: &'a Device,
    filter: ErrorFilter,
}

impl<'a> ErrorScope<'a> {
    pub fn new(device: &'a Device, filter: ErrorFilter) -> Self {
        device.push_error_scope(filter);
        Self { device, filter }
    }

    pub async fn check(self) -> Option<Error> {
        self.device.pop_error_scope().await
    }

    pub async fn execute<F, T, E>(self, operation: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
        E: From<Error>,
    {
        let result = operation();

        if let Some(error) = self.check().await {
            return Err(E::from(error));
        }

        result
    }
}

// Usage
async fn safe_operation_with_scope(device: &Device) -> Result<wgpu::Buffer, WGPUError> {
    let scope = ErrorScope::new(device, ErrorFilter::Validation);

    scope.execute(|| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scoped Buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(buffer)
    }).await
}
```

## Async Error Handling

### Safe Buffer Mapping

```rust
use futures_intrusive::channel::shared::oneshot_channel;
use std::time::Duration;

pub async fn safe_buffer_map_read(
    device: &Device,
    buffer: &wgpu::Buffer,
    timeout: Option<Duration>,
) -> Result<Vec<u8>, String> {
    // Validate buffer can be mapped
    if !buffer.usage().contains(wgpu::BufferUsages::MAP_READ) {
        return Err("Buffer does not support MAP_READ".to_string());
    }

    let (sender, receiver) = oneshot_channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // Handle timeout
    let map_result = if let Some(timeout_duration) = timeout {
        // Poll with timeout
        let start_time = std::time::Instant::now();
        loop {
            device.poll(wgpu::Maintain::Poll);
            if let Ok(Some(result)) = receiver.try_receive() {
                break Some(result);
            }
            if start_time.elapsed() > timeout_duration {
                return Err("Buffer mapping timed out".to_string());
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    } else {
        // Block until complete
        device.poll(wgpu::Maintain::Wait);
        receiver.receive().await
    };

    match map_result {
        Some(Ok(())) => {
            let data = buffer.slice(..).get_mapped_range();
            let result = data.to_vec();
            drop(data); // Release mapped range
            buffer.unmap();
            Ok(result)
        }
        Some(Err(error)) => {
            Err(handle_buffer_error(error))
        }
        None => {
            Err("Buffer mapping was cancelled".to_string())
        }
    }
}
```

### Retry Mechanisms

```rust
use std::time::Duration;
use tokio::time::sleep;

pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

pub async fn retry_with_backoff<F, T, E>(
    config: RetryConfig,
    operation: F,
) -> Result<T, E>
where
    F: Fn() -> Result<T, E>,
    E: std::fmt::Debug,
{
    let mut delay = config.base_delay;

    for attempt in 1..=config.max_attempts {
        match operation() {
            Ok(result) => return Ok(result),
            Err(error) => {
                if attempt == config.max_attempts {
                    return Err(error);
                }

                println!("Attempt {} failed: {:?}, retrying in {:?}", attempt, error, delay);
                sleep(delay).await;

                // Exponential backoff
                delay = std::cmp::min(
                    Duration::from_secs_f32(delay.as_secs_f32() * config.backoff_multiplier),
                    config.max_delay,
                );
            }
        }
    }

    unreachable!()
}

// Usage
async fn resilient_buffer_operation(device: &Device) -> Result<wgpu::Buffer, String> {
    retry_with_backoff(RetryConfig::default(), || {
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resilient Buffer"),
            size: 1024 * 1024, // 1MB
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // This would be async in real usage
        if let Some(error) = futures::executor::block_on(device.pop_error_scope()) {
            Err(format!("Buffer creation failed: {:?}", error))
        } else {
            Ok(buffer)
        }
    }).await
}
```

## Device and Adapter Error Handling

### Robust Device Creation

```rust
use wgpu::{
    Instance, RequestAdapterOptions, DeviceDescriptor, PowerPreference,
    Features, Limits, RequestDeviceError
};

pub async fn create_device_with_fallbacks() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = Instance::default();

    // Try high performance first
    let adapter_options = vec![
        RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        },
        RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        },
        RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: true, // Software fallback
        },
    ];

    for (i, options) in adapter_options.iter().enumerate() {
        println!("Trying adapter configuration {}", i + 1);

        let adapter = match instance.request_adapter(options).await {
            Some(adapter) => adapter,
            None => {
                println!("Adapter configuration {} failed", i + 1);
                continue;
            }
        };

        // Try different device configurations
        let device_configs = vec![
            DeviceDescriptor {
                label: Some("Full Features Device"),
                required_features: Features::all(),
                required_limits: Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::None,
            },
            DeviceDescriptor {
                label: Some("Basic Device"),
                required_features: Features::empty(),
                required_limits: Limits::downlevel_defaults(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::None,
            },
        ];

        for (j, config) in device_configs.iter().enumerate() {
            match adapter.request_device(config, None).await {
                Ok((device, queue)) => {
                    println!("Successfully created device with configuration {} and adapter {}", j + 1, i + 1);
                    return Ok((device, queue));
                }
                Err(RequestDeviceError::UnsupportedFeature(feature)) => {
                    println!("Device config {} failed: unsupported feature {:?}", j + 1, feature);
                }
                Err(RequestDeviceError::LimitsExceeded(limits)) => {
                    println!("Device config {} failed: limits exceeded {:?}", j + 1, limits);
                }
                Err(RequestDeviceError::DeviceLost(reason)) => {
                    println!("Device config {} failed: device lost {:?}", j + 1, reason);
                }
            }
        }
    }

    Err("Failed to create device with any configuration".to_string())
}
```

### Adapter Capability Checking

```rust
pub fn check_adapter_capabilities(adapter: &wgpu::Adapter) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let limits = adapter.limits();
    let features = adapter.features();

    // Check minimum requirements
    if limits.max_compute_workgroup_size_x < 64 {
        errors.push("Compute workgroup size too small".to_string());
    }

    if limits.max_storage_buffer_binding_size < 1024 * 1024 {
        errors.push("Storage buffer binding size too small".to_string());
    }

    if !features.contains(Features::TIMESTAMP_QUERY) {
        println!("Warning: Timestamp queries not supported");
    }

    if !features.contains(Features::BUFFER_BINDING_ARRAY) {
        println!("Warning: Buffer binding arrays not supported");
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
```

## Global Error Handlers

### Uncaptured Error Handler

```rust
use std::sync::Mutex;
use std::collections::VecDeque;

lazy_static::lazy_static! {
    static ref ERROR_LOG: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
}

pub fn setup_global_error_handler(device: &wgpu::Device) {
    device.on_uncaptured_error(Box::new(|error| {
        let error_msg = match error {
            Error::OutOfMemory { .. } => {
                "GPU out of memory - consider reducing buffer sizes or texture resolution".to_string()
            }
            Error::Validation { description, .. } => {
                format!("GPU validation error: {}", description)
            }
            Error::Internal { description, .. } => {
                format!("Internal GPU error: {}", description)
            }
        };

        // Log error
        eprintln!("Uncaptured GPU error: {}", error_msg);

        // Store in global error log
        if let Ok(mut log) = ERROR_LOG.lock() {
            log.push_back(error_msg);
            // Keep only last 100 errors
            while log.len() > 100 {
                log.pop_front();
            }
        }

        // Trigger error handling in application
        // trigger_error_recovery();
    }));
}

pub fn get_recent_errors() -> Vec<String> {
    ERROR_LOG.lock()
        .map(|log| log.iter().cloned().collect())
        .unwrap_or_default()
}
```

### Device Lost Recovery

```rust
use std::sync::{Arc, Mutex};

pub struct DeviceManager {
    device: Arc<Mutex<Option<wgpu::Device>>>,
    queue: Arc<Mutex<Option<wgpu::Queue>>>,
    recovery_callback: Option<Box<dyn Fn() + Send + Sync>>,
}

impl DeviceManager {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let manager = Self {
            device: Arc::new(Mutex::new(Some(device))),
            queue: Arc::new(Mutex::new(Some(queue))),
            recovery_callback: None,
        };

        // Setup device lost callback
        if let Ok(device_guard) = manager.device.lock() {
            if let Some(device) = device_guard.as_ref() {
                let device_arc = Arc::clone(&manager.device);
                let queue_arc = Arc::clone(&manager.queue);

                device.set_device_lost_callback(Box::new(move |reason, message| {
                    eprintln!("Device lost: {:?} - {}", reason, message);

                    // Clear device and queue
                    if let Ok(mut device_guard) = device_arc.lock() {
                        *device_guard = None;
                    }
                    if let Ok(mut queue_guard) = queue_arc.lock() {
                        *queue_guard = None;
                    }

                    // Trigger recovery
                    // self.trigger_recovery();
                }));
            }
        }

        manager
    }

    pub fn with_recovery_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.recovery_callback = Some(Box::new(callback));
        self
    }

    pub fn is_device_valid(&self) -> bool {
        self.device.lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    pub async fn recover_device(&self) -> Result<(), String> {
        // Attempt to recreate device
        match create_device_with_fallbacks().await {
            Ok((new_device, new_queue)) => {
                if let (Ok(mut device_guard), Ok(mut queue_guard)) =
                    (self.device.lock(), self.queue.lock()) {
                    *device_guard = Some(new_device);
                    *queue_guard = Some(new_queue);
                }

                // Trigger recovery callback
                if let Some(callback) = &self.recovery_callback {
                    callback();
                }

                Ok(())
            }
            Err(e) => Err(format!("Failed to recover device: {}", e))
        }
    }
}
```

## Error Reporting and Logging

### Structured Error Reporting

```rust
use serde::{Serialize, Deserialize};
use std::time::SystemTime;

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorReport {
    pub timestamp: SystemTime,
    pub error_type: String,
    pub description: String,
    pub context: Option<String>,
    pub recoverable: bool,
}

impl ErrorReport {
    pub fn from_wgpu_error(error: Error, context: Option<String>) -> Self {
        let (error_type, description, recoverable) = match error {
            Error::OutOfMemory { .. } => {
                ("OutOfMemory".to_string(), "GPU ran out of memory".to_string(), true)
            }
            Error::Validation { description, .. } => {
                ("Validation".to_string(), description, false)
            }
            Error::Internal { description, .. } => {
                ("Internal".to_string(), description, false)
            }
        };

        Self {
            timestamp: SystemTime::now(),
            error_type,
            description,
            context,
            recoverable,
        }
    }

    pub fn log_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        writeln!(file, "{}", json)?;
        Ok(())
    }
}
```

## Best Practices

1. **Use error scopes** for specific operations that might fail
2. **Handle device loss gracefully** with recovery mechanisms
3. **Implement retry logic** for transient failures
4. **Log errors comprehensively** for debugging
5. **Validate inputs early** before GPU operations
6. **Provide meaningful error messages** to users
7. **Test error conditions** in development
8. **Monitor error rates** in production

## Common Error Scenarios

### Memory Exhaustion

```rust
async fn handle_memory_exhaustion(device: &Device) -> Result<wgpu::Buffer, String> {
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Large Buffer"),
        size: 2 * 1024 * 1024 * 1024, // 2GB - likely to fail
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    if let Some(Error::OutOfMemory { .. }) = device.pop_error_scope().await {
        // Try smaller buffer
        println!("Large buffer allocation failed, trying smaller size");

        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let smaller_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Smaller Buffer"),
            size: 512 * 1024 * 1024, // 512MB
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        if device.pop_error_scope().await.is_some() {
            Err("Cannot allocate even small buffer - GPU memory exhausted".to_string())
        } else {
            Ok(smaller_buffer)
        }
    } else {
        Ok(buffer)
    }
}
```

### Validation Failures

```rust
fn handle_validation_error() -> Result<wgpu::Buffer, String> {
    // This will cause a validation error
    let invalid_descriptor = wgpu::BufferDescriptor {
        label: Some("Invalid Buffer"),
        size: 0, // Invalid: size cannot be 0
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    };

    // Validation happens immediately, not async
    // In real code, wrap in error scope for proper handling
    Err("Buffer size cannot be zero".to_string())
}
```