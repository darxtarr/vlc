//! Binary I/O for compressed indices
//! 
//! File layout:
//! - vlc.idx (manifest)
//! - vlc.anchors.bin (anchor vectors)
//! - vlc.assign.bin (assignments)
//! - vlc.jacobians.bin (optional)
//! - vlc.residuals.bin (optional)

use std::fs::File;
use std::io::{self, Read, Write, BufReader, BufWriter};
use std::path::Path;
use bytemuck;
use half::f16;

use crate::types::{CompressedIndex, AnchorSet, Assignments, IndexMetadata, QuantizationMode};
use crate::{VLC_MAGIC, VLC_VERSION};

/// Write compressed index to disk
pub fn write_index(index: &CompressedIndex, base_path: &str) -> io::Result<()> {
    let base = Path::new(base_path);
    
    // Write manifest file
    let manifest_path = base.with_extension("idx");
    let mut manifest = BufWriter::new(File::create(manifest_path)?);
    
    // Header
    manifest.write_all(&VLC_MAGIC.to_le_bytes())?;
    manifest.write_all(&VLC_VERSION.to_le_bytes())?;
    
    // Dimensions
    manifest.write_all(&(index.anchor_set.m as u32).to_le_bytes())?;
    manifest.write_all(&(index.anchor_set.d as u32).to_le_bytes())?;
    manifest.write_all(&(index.assignments.n as u32).to_le_bytes())?;
    
    // Flags
    let flags: u8 = 
        (index.anchor_set.jacobians.is_some() as u8) |
        ((index.assignments.residuals.is_some() as u8) << 1);
    manifest.write_all(&[flags])?;
    
    // Write anchors
    let anchors_path = base.with_extension("anchors.bin");
    let mut anchors_file = BufWriter::new(File::create(anchors_path)?);
    let anchor_bytes: &[u8] = bytemuck::cast_slice(&index.anchor_set.anchors);
    anchors_file.write_all(anchor_bytes)?;
    
    // Write assignments
    let assigns_path = base.with_extension("assign.bin");
    let mut assigns_file = BufWriter::new(File::create(assigns_path)?);
    let assign_bytes: &[u8] = bytemuck::cast_slice(&index.assignments.assign);
    assigns_file.write_all(assign_bytes)?;
    
    // Write optional Jacobians
    if let Some(ref jacobians) = index.anchor_set.jacobians {
        let jacobians_path = base.with_extension("jacobians.bin");
        let mut jacobians_file = BufWriter::new(File::create(jacobians_path)?);
        let jacobian_bytes: &[u8] = bytemuck::cast_slice(jacobians);
        jacobians_file.write_all(jacobian_bytes)?;
    }
    
    // Write optional residuals
    if let Some(ref residuals) = index.assignments.residuals {
        let residuals_path = base.with_extension("residuals.bin");
        let mut residuals_file = BufWriter::new(File::create(residuals_path)?);
        let residual_bytes: &[u8] = bytemuck::cast_slice(residuals);
        residuals_file.write_all(residual_bytes)?;
    }
    
    Ok(())
}

/// Read compressed index from disk
pub fn read_index(base_path: &str) -> io::Result<CompressedIndex> {
    let base = Path::new(base_path);
    
    // Read manifest
    let manifest_path = base.with_extension("idx");
    let mut manifest = BufReader::new(File::open(manifest_path)?);
    
    // Read and verify header
    let mut magic = [0u8; 4];
    manifest.read_exact(&mut magic)?;
    if u32::from_le_bytes(magic) != VLC_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid VLC magic number"
        ));
    }
    
    let mut version = [0u8; 2];
    manifest.read_exact(&mut version)?;
    if u16::from_le_bytes(version) != VLC_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported VLC version: {}", u16::from_le_bytes(version))
        ));
    }
    
    // Read dimensions
    let mut m_bytes = [0u8; 4];
    let mut d_bytes = [0u8; 4];
    let mut n_bytes = [0u8; 4];
    manifest.read_exact(&mut m_bytes)?;
    manifest.read_exact(&mut d_bytes)?;
    manifest.read_exact(&mut n_bytes)?;
    
    let m = u32::from_le_bytes(m_bytes) as usize;
    let d = u32::from_le_bytes(d_bytes) as usize;
    let n = u32::from_le_bytes(n_bytes) as usize;
    
    // Read flags
    let mut flags = [0u8; 1];
    manifest.read_exact(&mut flags)?;
    let has_jacobians = (flags[0] & 1) != 0;
    let has_residuals = (flags[0] & 2) != 0;
    
    // Read anchors
    let anchors_path = base.with_extension("anchors.bin");
    let anchors_data = std::fs::read(anchors_path)?;
    let anchors: Vec<f16> = bytemuck::cast_slice(&anchors_data).to_vec();
    
    // Read assignments
    let assigns_path = base.with_extension("assign.bin");
    let assigns_data = std::fs::read(assigns_path)?;
    let assign: Vec<u32> = bytemuck::cast_slice(&assigns_data).to_vec();
    
    // Read optional Jacobians
    let jacobians = if has_jacobians {
        let jacobians_path = base.with_extension("jacobians.bin");
        let jacobians_data = std::fs::read(jacobians_path)?;
        Some(bytemuck::cast_slice(&jacobians_data).to_vec())
    } else {
        None
    };
    
    // Read optional residuals
    let residuals = if has_residuals {
        let residuals_path = base.with_extension("residuals.bin");
        let residuals_data = std::fs::read(residuals_path)?;
        Some(bytemuck::cast_slice(&residuals_data).to_vec())
    } else {
        None
    };
    
    // Construct index
    let anchor_set = AnchorSet {
        anchors,
        jacobians,
        m,
        d,
    };
    
    let assignments = Assignments {
        assign,
        residuals,
        n,
        d_r: 0, // TODO: Store this in manifest
    };
    
    // Create placeholder metadata (should be stored in manifest)
    let metadata = IndexMetadata {
        n_original: n,
        d_original: d,
        compression_ratio: 0.0, // TODO: Calculate or store
        iterations: 0,
        final_energy: 0.0,
        has_jacobians,
        has_residuals,
        quantization: QuantizationMode::F16,
    };
    
    Ok(CompressedIndex {
        anchor_set,
        assignments,
        metadata,
    })
}

/// Get info about a compressed index without fully loading it
pub fn read_index_info(base_path: &str) -> io::Result<IndexMetadata> {
    let base = Path::new(base_path);
    let manifest_path = base.with_extension("idx");
    let mut manifest = BufReader::new(File::open(manifest_path)?);
    
    // Read header
    let mut magic = [0u8; 4];
    let mut version = [0u8; 2];
    manifest.read_exact(&mut magic)?;
    manifest.read_exact(&mut version)?;
    
    // Read dimensions
    let mut m_bytes = [0u8; 4];
    let mut d_bytes = [0u8; 4];
    let mut n_bytes = [0u8; 4];
    manifest.read_exact(&mut m_bytes)?;
    manifest.read_exact(&mut d_bytes)?;
    manifest.read_exact(&mut n_bytes)?;
    
    let m = u32::from_le_bytes(m_bytes) as usize;
    let d = u32::from_le_bytes(d_bytes) as usize;
    let n = u32::from_le_bytes(n_bytes) as usize;
    
    // Read flags
    let mut flags = [0u8; 1];
    manifest.read_exact(&mut flags)?;
    let has_jacobians = (flags[0] & 1) != 0;
    let has_residuals = (flags[0] & 2) != 0;
    
    // Calculate sizes
    let anchor_bytes = m * d * 2; // f16
    let assign_bytes = n * 4; // u32
    let jacobian_bytes = if has_jacobians { m * d * 2 } else { 0 };
    let residual_bytes = if has_residuals { n * d * 2 } else { 0 }; // Assuming full d for now
    
    let total_bytes = anchor_bytes + assign_bytes + jacobian_bytes + residual_bytes;
    let original_bytes = n * d * 4; // Assuming f32 originals
    
    Ok(IndexMetadata {
        n_original: n,
        d_original: d,
        compression_ratio: total_bytes as f32 / original_bytes as f32,
        iterations: 0,
        final_energy: 0.0,
        has_jacobians,
        has_residuals,
        quantization: QuantizationMode::F16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AnchorSet;
    
    #[test]
    fn test_roundtrip() {
        // Create dummy index
        let mut anchor_set = AnchorSet::new(2, 3);
        for i in 0..6 {
            anchor_set.anchors[i] = f16::from_f32(i as f32);
        }
        
        let mut assignments = Assignments::new(4);
        assignments.assign = vec![0, 1, 0, 1];
        
        let metadata = IndexMetadata {
            n_original: 4,
            d_original: 3,
            compression_ratio: 0.25,
            iterations: 100,
            final_energy: 0.5,
            has_jacobians: false,
            has_residuals: false,
            quantization: QuantizationMode::F16,
        };
        
        let index = CompressedIndex {
            anchor_set,
            assignments,
            metadata,
        };
        
        // Write and read back
        let temp_path = "/tmp/test_vlc";
        write_index(&index, temp_path).unwrap();
        let loaded = read_index(temp_path).unwrap();
        
        // Verify
        assert_eq!(loaded.anchor_set.m, 2);
        assert_eq!(loaded.anchor_set.d, 3);
        assert_eq!(loaded.assignments.n, 4);
        assert_eq!(loaded.assignments.assign, vec![0, 1, 0, 1]);
        
        // Clean up
        std::fs::remove_file(format!("{}.idx", temp_path)).ok();
        std::fs::remove_file(format!("{}.anchors.bin", temp_path)).ok();
        std::fs::remove_file(format!("{}.assign.bin", temp_path)).ok();
    }
}
