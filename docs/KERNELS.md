# GPU Kernel Specifications

**Implementation Status**: All core kernels (K1-K3) are FULLY IMPLEMENTED and CODE-COMPLETE as of M2 milestone (98% complete, pending hardware validation). See `/home/u/code/vlc/src/gpu/` for implementation.

## Memory Layout Conventions

All matrices are **row-major** (point's dimensions are contiguous).
All buffers are **16-byte aligned** for optimal GPU access.

## Kernel 1: Point Assignment (IMPLEMENTED)

**Status**: Fully implemented in `src/gpu/shaders/assign.wgsl` and `src/gpu/ops.rs::assign_points()`

### Purpose
Assign each point to its nearest anchor using L2 distance.

### Interface
```wgsl
// Inputs
@group(0) @binding(0) var<storage, read> points: array<f32>;     // [n × d]
@group(0) @binding(1) var<storage, read> anchors: array<f32>;    // [m × d]
@group(0) @binding(2) var<storage, read_write> assigns: array<u32>; // [n]

// Uniforms
struct Params {
    n: u32,  // number of points
    m: u32,  // number of anchors
    d: u32,  // dimension
}
@group(1) @binding(0) var<uniform> params: Params;
```

### Algorithm
```
WORKGROUP_SIZE = 256
Each thread handles one point

thread_id = global_invocation_id.x
if thread_id >= n: return

point_offset = thread_id * d
min_dist = INFINITY
min_anchor = 0

for anchor_id in 0..m:
    dist = 0.0
    anchor_offset = anchor_id * d
    
    // Unrolled inner loop for better performance
    for dim in 0..d step 4:
        diff0 = points[point_offset + dim + 0] - anchors[anchor_offset + dim + 0]
        diff1 = points[point_offset + dim + 1] - anchors[anchor_offset + dim + 1]
        diff2 = points[point_offset + dim + 2] - anchors[anchor_offset + dim + 2]
        diff3 = points[point_offset + dim + 3] - anchors[anchor_offset + dim + 3]
        dist += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
    
    if dist < min_dist:
        min_dist = dist
        min_anchor = anchor_id

assigns[thread_id] = min_anchor
```

### Optimizations
- Cache anchors in shared memory if m is small
- Use warp-level primitives for reduction
- Consider tiling for very large d

## Kernel 2: Robust Reduction (IMPLEMENTED)

**Status**: Fully implemented in `src/gpu/shaders/reduce.wgsl` and `src/gpu/ops.rs::reduce_stats()`

### Purpose
Compute robust mean and statistics for points assigned to each anchor.

### Interface
```wgsl
// Inputs
@group(0) @binding(0) var<storage, read> points: array<f32>;     // [n × d]
@group(0) @binding(1) var<storage, read> assigns: array<u32>;    // [n]
@group(0) @binding(2) var<storage, read_write> stats: array<f32>; // [m × (d+2)]
// stats layout: [mean_0..mean_d-1, count, variance]

// Uniforms include trim_percent for robust mean
struct Params {
    n: u32,
    m: u32, 
    d: u32,
    trim_percent: f32,  // e.g., 0.1 for 10% trimming
    huber_threshold: f32,
}
```

### Algorithm (Two-Pass)
```
WORKGROUP_SIZE = 256
Each workgroup handles one anchor

anchor_id = workgroup_id.x
thread_id = local_invocation_id.x

// Shared memory for reduction
var<workgroup> partial_sums[256 * d];
var<workgroup> point_count: atomic<u32>;
var<workgroup> distances[256];  // for robust estimation

// Pass 1: Collect points and distances
local_count = 0
for point_id in thread_id..n step WORKGROUP_SIZE:
    if assigns[point_id] == anchor_id:
        // Store distance for sorting
        distances[local_count] = compute_distance(point_id, anchor_id)
        local_count += 1

// Workgroup barrier
workgroupBarrier()

// Sort distances (bitonic sort in shared memory)
bitonic_sort(distances, local_count)

// Determine trim boundaries
trim_count = u32(local_count * trim_percent)
low_cutoff = distances[trim_count]
high_cutoff = distances[local_count - trim_count - 1]

// Pass 2: Accumulate trimmed mean
for point_id in thread_id..n step WORKGROUP_SIZE:
    if assigns[point_id] == anchor_id:
        dist = compute_distance(point_id, anchor_id)
        if dist >= low_cutoff && dist <= high_cutoff:
            // Accumulate point
            for dim in 0..d:
                atomicAdd(&partial_sums[thread_id * d + dim], 
                         points[point_id * d + dim])
            atomicAdd(&point_count, 1)

// Tree reduction of partial sums
tree_reduce(partial_sums, point_count)

// Thread 0 writes final stats
if thread_id == 0:
    for dim in 0..d:
        stats[anchor_id * (d+2) + dim] = partial_sums[dim] / f32(point_count)
    stats[anchor_id * (d+2) + d] = f32(point_count)
    stats[anchor_id * (d+2) + d + 1] = variance  // computed similarly
```

### Alternative: Huber Mean
Instead of trimming, use Huber weighting:
```
weight = if dist < huber_threshold:
    1.0
else:
    huber_threshold / dist
```

## Kernel 3: Anchor Update (IMPLEMENTED)

**Status**: Fully implemented in `src/gpu/shaders/update.wgsl` and `src/gpu/ops.rs::update_anchors()`

### Purpose
Update anchor positions using computed statistics and temperature.

### Interface
```wgsl
@group(0) @binding(0) var<storage, read_write> anchors: array<f32>;  // [m × d]
@group(0) @binding(1) var<storage, read> stats: array<f32>;         // [m × (d+2)]
@group(0) @binding(2) var<storage, read_write> jacobians: array<f32>; // [m × d] optional

struct Params {
    m: u32,
    d: u32,
    temperature: f32,
    learning_rate: f32,
    momentum: f32,
    enable_jacobians: u32,
}
```

### Algorithm
```
WORKGROUP_SIZE = 256
Each thread handles one dimension of one anchor

global_id = global_invocation_id.x
anchor_id = global_id / d
dim_id = global_id % d

if anchor_id >= m: return

// Read statistics
mean_value = stats[anchor_id * (d+2) + dim_id]
point_count = stats[anchor_id * (d+2) + d]

// Skip if no points assigned
if point_count < 1.0: return

// Current anchor value
old_value = anchors[anchor_id * d + dim_id]

// Gradient step with temperature scaling
step = (mean_value - old_value) * learning_rate * (temperature / (1.0 + temperature))

// Apply momentum if specified
if momentum > 0.0:
    step = momentum * previous_step[anchor_id * d + dim_id] + (1-momentum) * step

// Update anchor (with clamping for stability)
new_value = old_value + clamp(step, -0.1, 0.1)
anchors[anchor_id * d + dim_id] = new_value

// Optional: Update diagonal Jacobian
if enable_jacobians == 1:
    // Simple exponential moving average of local gradients
    local_variance = stats[anchor_id * (d+2) + d + 1]
    jacobians[anchor_id * d + dim_id] = 
        0.9 * jacobians[anchor_id * d + dim_id] + 
        0.1 * sqrt(local_variance + 0.001)  // regularization
```

## Kernel 4: Maintenance - Merge (NOT IMPLEMENTED - M3)

**Status**: Planned for M3 milestone, not yet implemented

### Purpose
Merge anchors that are closer than threshold.

### Algorithm (CPU orchestrated, GPU assisted)
```
1. GPU: Compute pairwise distances between anchors (upper triangular)
2. CPU: Find pairs below threshold, create merge list
3. GPU: For each merge pair, weighted average based on point counts
4. GPU: Reassign points from merged anchors
```

## Kernel 5: Maintenance - Split (NOT IMPLEMENTED - M3)

**Status**: Planned for M3 milestone, not yet implemented

### Purpose
Split anchors with too many assigned points.

### Algorithm
```
For each overloaded anchor:
1. Find principal component of assigned points (power iteration)
2. Create two new anchors along principal axis
3. Reassign points to nearest of the two
```

## Memory Bandwidth Optimization

### Tiling Strategy
For large d (e.g., 768), tile the computation:
```
TILE_SIZE = 128
for tile_start in 0..d step TILE_SIZE:
    // Load tile into shared memory
    // Compute partial distances
    // Accumulate
```

### Coalescing Rules
- Consecutive threads access consecutive memory
- Align all buffers to 128 bytes (cache line)
- Use vector loads (float4) where possible

## Error Handling

Each kernel should include:
```wgsl
// Bounds checking
if thread_id >= n: return

// NaN/Inf detection
if !is_finite(value):
    value = 0.0  // or other safe default
    
// Atomic operation validation
loop_guard = 0
while atomicCompareExchangeWeak(&lock, 0u, 1u) == 1u:
    loop_guard += 1
    if loop_guard > 1000:
        return  // Prevent infinite spin
```

## Performance Targets

**Note**: These are aspirational targets. Actual performance validation pending GPU hardware access.

- Assignment: 1ms for 1M points, 4K anchors, d=768 (target)
- Reduction: 2ms for same (target)
- Update: 0.5ms for 4K anchors, d=768 (target)
- Overall iteration: <5ms on RTX 3080 (target)

**Current Status**: Implementation complete, benchmarking requires GPU hardware validation (see STATUS.md)

## Testing Each Kernel

**Implementation Status**: Test infrastructure complete with `test-gpu` command in `src/bin/vlc.rs`

Available tests:
1. `cargo run --bin vlc test-gpu` - GPU vs CPU comparison on small dataset
2. `cargo run --bin vlc test-gpu --large` - GPU vs CPU on large dataset (10K points)

Planned tests (not yet implemented):
1. Edge cases (0 assignments, all same anchor, etc.)
2. Numerical stability verification with random data
3. Memory bounds checking with compute sanitizer

## Note on f16 vs f32

Start with f32 for correctness, then optimize:
- Store anchors as f16
- Compute distances in f32
- Accumulate in f32
- Write back as f16

This prevents numerical drift while saving memory bandwidth.
