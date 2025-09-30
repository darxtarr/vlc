# Implementation Guide for Sonnet

## Project Philosophy
- We are a code boutique, not a factory
- No unnecessary dependencies - recreate what we need
- Binary formats over JSON in hot paths
- Understand every byte, every operation

## Phase 1: Foundation (Start Here)

### Step 1: Create Rust Project Structure
```
cargo new --lib vlc
cd vlc
```

### Step 2: Minimal Cargo.toml
```toml
[package]
name = "vlc"
version = "0.1.0"
edition = "2021"

[dependencies]
# ONLY these, no more without explicit discussion
candle-core = "0.3"  # For CPU math
wgpu = "0.19"        # For GPU kernels
bytemuck = "1.14"    # For safe transmutes
half = "2.3"         # For f16 support

[dev-dependencies]
# For testing only
rand = "0.8"
```

### Step 3: Core Data Structures
Create `src/types.rs`:
- Define `Anchor`, `Assignment`, `CompressedIndex` structs
- Use `[f16]` for anchor storage, not Vec<Vec<f32>>
- Align structures to 16 bytes for GPU

### Step 4: Binary I/O Module
Create `src/io.rs`:
- Implement readers/writers for binary formats
- Use mmap for large files
- Include magic numbers and version headers
- NO serde, NO JSON in critical path

## Phase 2: CPU Prototype (M1)

### Step 5: Basic Operations
Create `src/ops/cpu.rs`:
```rust
// Core functions to implement:
fn assign_points(points: &[f16], anchors: &[f16], d: usize) -> Vec<u32>
fn compute_robust_mean(points: &[f16], assignments: &[u32], anchor_id: u32) -> Vec<f16>
fn update_anchor(anchor: &mut [f16], mean: &[f16], temperature: f32)
```

### Step 6: Annealing Loop
Create `src/anneal.rs`:
- Simple convergence loop
- Energy calculation (distortion only for M1)
- Temperature schedule (exponential cooling)
- Log only: iteration, temperature, energy, assignment_changes

### Step 7: Testing Harness
Create `tests/synthetic.rs`:
- Generate Gaussian blob data (3 clusters, 1000 points, d=128)
- Run compression
- Verify: assignments are stable, energy decreases

## Phase 3: GPU Kernels (M2)

### Step 8: WGPU Setup
Create `src/gpu/context.rs`:
- Initialize WGPU device/queue
- Create pipeline layouts
- Handle buffer management

### Step 9: Assign Kernel
Create `src/gpu/kernels/assign.wgsl`:
```wgsl
// Pseudocode structure:
@compute @workgroup_size(256)
fn assign_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    // bind groups for points, anchors, assignments
) {
    let point_idx = gid.x;
    if (point_idx >= n_points) { return; }
    
    // Load point
    // Find nearest anchor (loop over m)
    // Write assignment
}
```

### Step 10: Reduce Kernel  
Create `src/gpu/kernels/reduce.wgsl`:
- Tree reduction pattern
- Atomic accumulation into shared memory
- Robust mean with trimming or Huber

### Step 11: Update Kernel
Create `src/gpu/kernels/update.wgsl`:
- Read statistics
- Apply gradient step scaled by temperature
- Optional: update diagonal Jacobian

## Phase 4: Maintenance & Retrieval (M3)

### Step 12: Maintenance Operations
Create `src/maintenance.rs`:
- Merge: combine anchors < threshold apart
- Split: divide anchors with > threshold assignments
- Quantize: snap to grid (int8 or lower precision)

### Step 13: Compressed Retrieval
Create `src/retrieval.rs`:
- Find top-k anchors (small, can be brute force)
- Gather assigned points
- Reconstruct with residuals/Jacobians
- Return final top-k

### Step 14: CLI Interface
Create `src/bin/vlc.rs`:
```rust
// Commands:
// vlc index --emb <path> --d <dim> --m <anchors> --out <path>
// vlc eval --idx <path> --queries <path> --k <k>
// vlc info --idx <path>  // Print compression stats
```

## Critical Implementation Notes

### Memory Alignment
- GPU requires 16-byte alignment for buffers
- Use `#[repr(C, align(16))]` for structures
- Pad arrays to multiple of workgroup size

### Numerical Stability
- Convert f16→f32 for accumulation operations
- Use Kahan summation for large reductions
- Clamp gradients to prevent explosion

### Performance Patterns
- Coalesced memory access (consecutive threads → consecutive memory)
- Minimize divergence (all threads in warp take same branch)
- Use shared memory for repeated anchor access

### Debugging Strategy
1. Start with tiny data (10 points, 2 anchors)
2. Print every intermediate state
3. Verify CPU and GPU produce identical results
4. Only then scale up

## What NOT to Do
- Don't use ndarray, nalgebra, or other linalg crates
- Don't implement generic N-dimensional tensors
- Don't add progress bars or fancy logging
- Don't optimize prematurely - correctness first

## Validation Checklist
- [ ] Binary files are readable with `xxd`
- [ ] Can round-trip: data → compress → decompress → data
- [ ] GPU kernels match CPU implementation exactly
- [ ] Memory usage is predictable: O(m*d + n)
- [ ] No allocations in hot loops

## Questions for Human Before Starting
1. Should we use workgroup size 64, 128, or 256?
2. Do you want checkpointing during training?
3. Should we implement early stopping based on validation set?

## The Prime Directive
Every line of code should be intentional. If you don't understand why something is there, it shouldn't be there. This is boutique craftsmanship, not factory assembly.
