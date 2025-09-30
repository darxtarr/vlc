# VLC Design Document

## System Architecture

### Overview
Vector-Lattice Compression (VLC) learns an adaptive tessellation of embedding space through annealed optimization. GPU-accelerated via WGPU, orchestrated by Rust.

## Core Components

### 1. Data Structures

```rust
// Conceptual structures (not final Rust)
struct AnchorSet {
    anchors: Vec<f16>,     // [m × d] row-major
    jacobians: Option<Vec<f16>>, // [m × d] diagonal only
    temperature: f32,       // annealing state
}

struct PointAssignments {
    assign: Vec<u32>,      // [n] -> anchor indices
    residuals: Option<Vec<f16>>, // [n × d_r] compressed
}
```

### 2. GPU Kernels (WGPU)

#### K1: Assign Kernel
- **Input**: Points X[n×d], Anchors A[m×d]
- **Output**: Assignments assign[n]
- **Logic**: For each point, find nearest anchor (L2)
- **Parallelism**: One thread per point, tiled for cache
- **Memory**: Coalesced reads of X, broadcast reads of A

#### K2: Reduce Kernel  
- **Input**: Points X[n×d], Assignments assign[n]
- **Output**: Stats per anchor (sum, count, variance)
- **Logic**: Parallel reduction with robust mean (trim outliers or Huber)
- **Parallelism**: Tree reduction per anchor
- **Key**: Atomic adds for accumulation, f16→f32 promotion

#### K3: Update Kernel
- **Input**: Stats, learning rate, temperature
- **Output**: Updated anchors A'[m×d], optional J'[m×d]
- **Logic**: Move anchors toward robust mean, update Jacobians via mini-LS
- **Constraint**: Clipped steps to prevent overshooting

#### K4: Maintenance Kernel Suite
- **Merge**: Combine anchors closer than threshold
- **Split**: Divide overloaded anchors (too many assignments)
- **Quantize**: Snap to reduced precision grid
- **Topology**: Sample triplets, compute margin loss

### 3. Annealing Schedule

```
T(t) = T_0 * exp(-λt)  // Exponential cooling
OR
T(t) = T_0 / (1 + λt)  // Inverse cooling
```

Tie to Huber threshold: `τ(T) = κ * sqrt(T)` - wide tolerance when hot, precise when cool.

### 4. Energy Function

```
E_total = E_distortion + λ_size * E_footprint + λ_topo * E_topology

E_distortion = Σ_i ||x_i - reconstruct(x_i)||²
E_footprint = sizeof(anchors + assignments + residuals)  
E_topology = Σ_triplets margin_loss(q, p, n)
```

### 5. Binary Layout Specifications

#### anchors.bin
```
[header: 8 bytes]
  - magic: u32 = 0x564C4341  // "VLCA"
  - version: u16 = 1
  - m: u16 (num anchors)
[data: m*d*2 bytes]  
  - f16 values in row-major order
```

#### assign.bin
```
[header: 8 bytes]
  - magic: u32 = 0x564C4341  // "VLCA"  
  - version: u16 = 1
  - n: u16 (num points, lower 16 bits)
  - n_high: u16 (upper 16 bits)
[data: n*4 bytes]
  - u32 anchor indices
```

## Algorithm Flow

### Training Phase

```
1. Initialize anchors (k-means++ or random subset)
2. Set T = T_0 (high temperature)
3. While not converged:
   a. Assign points to nearest anchors
   b. Compute robust statistics per anchor
   c. Update anchors (gradient step scaled by T)
   d. If enabled, update Jacobians
   e. Every P iterations:
      - Merge close anchors
      - Split overloaded anchors  
      - Quantize if late in schedule
   f. Cool temperature
   g. Check convergence (energy plateau or recall target)
```

### Retrieval Phase

```
1. Encode query q
2. Find top-k nearest anchors (brute force over m)
3. Gather points assigned to those anchors
4. Reconstruct points:
   ŷ_i = a_j + J_j * (x_i - a_j) + r_i
5. Compute exact distances to query
6. Return top-k by distance
```

## Hyperparameters

- `m`: Number of anchors (typically 0.01n to 0.1n)
- `T_0`: Initial temperature (1.0 to 10.0)
- `λ_cool`: Cooling rate (0.01 to 0.1)
- `λ_size`: Footprint penalty (tune for target compression)
- `trim_pct`: Outlier trimming for robust mean (0.05 to 0.1)
- `merge_thresh`: Distance threshold for anchor merging
- `split_thresh`: Assignment count threshold for splitting

## Failure Modes & Mitigations

### Mode Collapse
- **Symptom**: All points assign to few anchors
- **Detection**: Monitor assignment histogram entropy
- **Mitigation**: Increase m, add entropy injection, slower cooling

### Quantization Shock  
- **Symptom**: Sudden recall drop when quantizing
- **Detection**: Track energy discontinuity
- **Mitigation**: Gradual quantization, mixed precision for hot anchors

### Jacobian Overfit
- **Symptom**: Good training recall, poor test recall
- **Detection**: Train/test divergence
- **Mitigation**: L2 regularization, early stop, diagonal-only constraint

## Performance Targets

- Compression: ≤30% of original size
- Recall@10: ≥0.95 vs baseline HNSW
- Training: <1 hour for 1M vectors on consumer GPU
- Query: <1ms per query on GPU, <10ms on CPU

## Testing Strategy

### Synthetic Tests
- Gaussian blobs: Known cluster structure
- Grid lattice: Perfect regular structure  
- Manifold data: Swiss roll, sphere surface

### Real Data
- Text embeddings: High intrinsic dimension
- Image embeddings: Natural cluster hierarchy
- Code embeddings: Discrete semantic regions

### Metrics
- Recall@k for k ∈ {1, 5, 10, 50}
- Compression ratio (bytes_vlc / bytes_original)
- Query latency (p50, p95, p99)
- Assignment churn rate during training

## Notes for Implementation

1. **Determinism**: Fix all random seeds for reproducibility
2. **Logging**: Scalar metrics only, no verbose output
3. **Checkpointing**: Save anchors every 100 iterations
4. **Validation**: Hold out 10% of data for convergence checking
5. **Memory**: Stream large datasets, don't load all at once

## Key Insight

We're not just compressing vectors - we're discovering the grammar of embedding space. The anchors form consonants, residuals are vowels, Jacobians provide inflection. Together they reconstruct meaning with minimal information.
