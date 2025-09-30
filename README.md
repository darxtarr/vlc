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
- `*.anchors.bin`: m×d f16 row-major anchor vectors
- `*.assign.bin`: n×u32 point-to-anchor assignments
- `*.jacobians.bin`: (optional) m×d f16 diagonal Jacobians
- `*.residuals.bin`: (optional) n×d_r f16/int8 residuals
- `*.idx`: manifest with magic number and metadata

## Implementation Status

- ✅ **M1: CPU prototype** - Complete, 2-3% compression, all tests passing
- 🏗️ **M2: GPU kernels** - Architecture complete, API integration needed
- ❌ **M3: Maintenance ops** - Not started (merge/split/quantize)

## Quick Start

```bash
# Build
cargo build --release

# Test with synthetic data
cargo run --bin vlc test

# View compressed index info
cargo run --bin vlc info --idx ./test_vlc

# Build index (real data loader not yet implemented)
# vlc index --emb embeddings.bin --d 768 --m 4096 --out ./vlc_idx
```

## Performance

**Current (M1 CPU)**:
- Compression: **2-3%** (10x better than 30% target!)
- Small test (300×64D): <1s, 3.23% compression
- Large test (10K×128D): 110s, 2.06% compression

**Expected (M2 GPU)**:
- 5-10x speedup on medium datasets
- <20min for 1M vectors

## Key Insights

This isn't just compression - it's learning the manifold structure of embedding space. The anchors discover the natural crystalline structure of the data, spending bits where meaning lives.

## Dependencies

Minimal and intentional:
```toml
candle-core = "0.9"          # CPU math operations
wgpu = "26.0"                # GPU compute kernels
bytemuck = "1.19"            # Zero-copy type conversion
half = "2.4"                 # f16 support
memmap2 = "0.9"              # Memory-mapped file I/O
futures-intrusive = "0.5"    # Async GPU operations
```

## Documentation

- `STATUS.md` - Current implementation status
- `docs/DESIGN.md` - System architecture
- `docs/KERNELS.md` - GPU kernel specifications
- `docs/M2_HANDOVER.md` - GPU implementation guide
- `docs/SONNET_GUIDE.md` - Implementation reference
- `docs/wgpu-reference/` - WGPU API documentation

## Testing

```bash
cargo test              # Run unit tests (all passing)
cargo run --bin vlc test      # Synthetic compression test
cargo build --release   # Production build
```

## Project Structure

```
vlc/
├── src/
│   ├── types.rs           # Core data structures
│   ├── anneal.rs          # Annealing loop (CPU + GPU)
│   ├── io.rs              # Binary I/O
│   ├── ops/cpu.rs         # CPU operations
│   ├── gpu/               # GPU acceleration
│   │   ├── context.rs     # WGPU setup
│   │   ├── ops.rs         # GPU operations
│   │   └── shaders/       # WGSL compute kernels
│   └── bin/vlc.rs         # CLI interface
├── docs/                  # Documentation
└── tests/                 # Unit tests
```

## License

MIT

---

*Boutique code, boutique results* 💎
