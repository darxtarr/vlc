# VLC Implementation Handover

## Quick Context
Vector-Lattice Compression: Learn Voronoi tessellation of embedding space. Compress to 30% size, maintain 95% recall.
**Philosophy**: Boutique code. No frameworks. Understand every byte.

## What Works (Don't Touch)
- `types.rs`: Data structures ✅ (f16 storage, GPU alignment)
- `ops/cpu.rs`: CPU kernels ✅ (assign, reduce, update)  
- `anneal.rs`: Training loop ✅ (needs I/O to actually run)
- Docs: All guidance solid

## Critical Missing Pieces (Implement These)

### 1. Fix lib.rs (5 min)
```rust
// Remove these lines (modules don't exist yet):
// pub mod io;
// pub mod gpu;
// pub use io::{read_index, write_index};
```

### 2. Create src/io.rs (30 min)
```rust
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Write, Read};

pub fn write_index(index: &CompressedIndex, path: &str) -> Result<()> {
    // Header: magic (u32) + version (u16) + m (u32) + d (u32) + n (u32)
    // Then: anchors.bin, assign.bin as separate files
    // Use bytemuck::cast_slice for zero-copy f16→u8
}

pub fn read_index(path: &str) -> Result<CompressedIndex> {
    // Mmap files, cast back to types
    // Validate magic number 0x564C4341
}
```

### 3. Create src/bin/vlc.rs (20 min)
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Index { 
        #[arg(long)] emb: String,
        #[arg(long)] d: usize,
        #[arg(long)] m: usize,
        #[arg(long)] out: String,
    },
    Info {
        #[arg(long)] idx: String,
    },
}

fn main() {
    // Parse args, dispatch to compress() or print info
}
```

### 4. Create tests/synthetic.rs (15 min)
```rust
// Generate Gaussian blobs
fn make_blobs(n: usize, d: usize, centers: usize) -> Vec<f16> {
    // Use rand::distributions::Normal
    // Create k centers randomly
    // Generate n/k points around each center
}

#[test]
fn test_compress_blobs() {
    let data = make_blobs(1000, 128, 3);
    let compressed = vlc::compress(&data, 1000, 128, Default::default());
    assert!(compressed.metadata.compression_ratio < 0.3);
}
```

### 5. Add k-means++ init in anneal.rs (10 min)
Replace random init with:
```rust
fn initialize_anchors() {
    // 1. Pick first anchor randomly
    // 2. For each next anchor:
    //    - Compute D(x)² for each point (distance to nearest anchor)
    //    - Sample next anchor proportional to D(x)²
}
```

## GPU Implementation (Phase 2 - After Above Works)

### Key WGPU Setup Pattern
```rust
// src/gpu/context.rs
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    assign_pipeline: wgpu::ComputePipeline,
}

// Create buffers with proper usage flags
let buffer = device.create_buffer(&wgpu::BufferDescriptor {
    size: (n * d * 4) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

### WGSL Kernel Structure
```wgsl
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> anchors: array<f32>;  
@group(0) @binding(2) var<storage, read_write> assigns: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }
    // Kernel logic here
}
```

## Critical Numbers
- Magic: `0x564C4341` ("VLCA")
- Workgroup size: 256 (start here, tune later)
- f16 → f32 for compute, f32 → f16 for storage
- Align buffers to 16 bytes

## Testing Order
1. Test CPU ops individually (already has tests)
2. Test I/O round-trip (write → read → compare)
3. Test full compression on tiny data (10 points, 2 anchors)
4. Test synthetic blobs
5. Scale up gradually

## Gotchas
- Don't use ndarray/nalgebra - raw slices only
- Check bounds in kernels (idx >= n)
- Use `bytemuck::cast_slice` for zero-copy type conversion
- Temperature affects convergence dramatically - start with T=1.0

## Success Metrics
- M1: CPU compress runs, <30% compression on blobs
- M2: GPU 10x faster than CPU
- M3: Recall@10 > 0.95 on real embeddings

## Debug If Stuck
```bash
# Check binary format
xxd -l 64 anchors.bin

# Print first/last anchor
cargo run -- info --idx ./test_idx

# Verify assignments histogram  
cargo test -- --nocapture
```

---
*Handover complete. Start with items 1-4 to get basic system running. Philosophy: think twice, code once.*
