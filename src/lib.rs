//! Vector-Lattice Compression
//! 
//! A boutique implementation of adaptive embedding compression through learned Voronoi tessellation.
//! No frameworks, no magic - just intentional, comprehensible code.

pub mod types;
pub mod io;
pub mod ops;
pub mod anneal;
pub mod gpu;
pub mod retrieval;

// Re-export core types
pub use types::{CompressedIndex, AnchorSet, Assignments};
pub use io::{read_index, write_index};
pub use anneal::{AnnealingConfig, compress};

#[cfg(feature = "gpu-wgpu")]
pub use anneal::compress_gpu;

#[cfg(feature = "gpu-cuda")]
pub use anneal::compress_cuda;

/// Our magic number for file headers
pub const VLC_MAGIC: u32 = 0x564C4341; // "VLCA" in hex

/// Version for binary format compatibility
pub const VLC_VERSION: u16 = 1;