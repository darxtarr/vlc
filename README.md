# Vector-Lattice Compression (VLC)

A boutique implementation of adaptive embedding compression using learned Voronoi tessellation.

## Philosophy

We are a code boutique, not a code factory. 
- No pulled dependencies where we can surgically recreate what's needed
- Frameworks are avoided in favor of understanding
- Every byte is accounted for, every operation intentional
- Think twice, code once

## Core Concept

Compress embedding vectors to ≤30% of original size while maintaining ≥95% recall@k.

Instead of storing every vector, we:
1. Learn representative anchors (family portraits)
2. Store only assignment indices and small residuals (sticky notes about differences)
3. Optionally learn local linear approximations (Jacobians) for better reconstruction

## Architecture

Pure Rust with WGPU for GPU kernels and Candle for host-side math. No frameworks, no hidden magic.

### Binary Formats
- `anchors.bin`: m×d f16 row-major anchor vectors
- `assign.bin`: n×u32 point-to-anchor assignments  
- `J.bin`: (optional) m×d f16 diagonal Jacobians
- `residuals.bin`: (optional) n×d_r f16/int8 residuals
- `meta.json`: inspection only, not used in hot path

## Implementation Status

- [ ] M1: CPU prototype with basic Assign/Reduce/Update
- [ ] M2: WGPU kernels + annealing loop + binary I/O
- [ ] M3: Maintenance ops (merge/split) + compressed retrieval

## Quick Start

```bash
# Build index
vlc index --emb embeddings.bin --d 768 --m 4096 --out ./vlc_idx

# Evaluate  
vlc eval --idx ./vlc_idx --queries queries.bin --baseline hnsw.idx --k 10
```

## Key Insights

This isn't just compression - it's learning the manifold structure of embedding space. The anchors discover the natural crystalline structure of the data, spending bits where meaning lives.
