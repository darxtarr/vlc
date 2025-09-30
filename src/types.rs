//! Core data structures for VLC
//! 
//! All structures are designed for zero-copy operations and GPU compatibility.
//! We use f16 (half) for storage and f32 for computation.

use half::f16;

/// Complete compressed index containing all data needed for reconstruction
#[derive(Debug)]
pub struct CompressedIndex {
    pub anchor_set: AnchorSet,
    pub assignments: Assignments,
    pub metadata: IndexMetadata,
}

/// Set of anchor points with optional Jacobians for local linear approximation
#[repr(C, align(16))]  // GPU alignment
#[derive(Debug)]
pub struct AnchorSet {
    /// Anchor vectors [m × d] in row-major order
    pub anchors: Vec<f16>,
    
    /// Optional diagonal Jacobians [m × d] for linear approximation
    pub jacobians: Option<Vec<f16>>,
    
    /// Number of anchors
    pub m: usize,
    
    /// Dimension of each anchor
    pub d: usize,
}

impl AnchorSet {
    /// Create a new anchor set with given capacity
    pub fn new(m: usize, d: usize) -> Self {
        Self {
            anchors: vec![f16::ZERO; m * d],
            jacobians: None,
            m,
            d,
        }
    }
    
    /// Get anchor at index as slice
    pub fn get_anchor(&self, idx: usize) -> &[f16] {
        let start = idx * self.d;
        &self.anchors[start..start + self.d]
    }
    
    /// Get mutable anchor at index
    pub fn get_anchor_mut(&mut self, idx: usize) -> &mut [f16] {
        let start = idx * self.d;
        let end = start + self.d;
        &mut self.anchors[start..end]
    }
}

/// Point-to-anchor assignments with optional residuals
#[repr(C, align(16))]
#[derive(Debug, Clone)]
pub struct Assignments {
    /// Assignment of each point to an anchor [n]
    pub assign: Vec<u32>,
    
    /// Optional residuals for better reconstruction [n × d_r]
    pub residuals: Option<Vec<f16>>,
    
    /// Number of points
    pub n: usize,
    
    /// Residual dimension (may be less than full d)
    pub d_r: usize,
}

impl Assignments {
    /// Create new assignments for n points
    pub fn new(n: usize) -> Self {
        Self {
            assign: vec![0; n],
            residuals: None,
            n,
            d_r: 0,
        }
    }
    
    /// Count points assigned to each anchor
    pub fn count_per_anchor(&self, m: usize) -> Vec<usize> {
        let mut counts = vec![0; m];
        for &a in &self.assign {
            counts[a as usize] += 1;
        }
        counts
    }
}

/// Metadata for compressed index
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// Original number of vectors
    pub n_original: usize,
    
    /// Original dimension
    pub d_original: usize,
    
    /// Compression ratio achieved
    pub compression_ratio: f32,
    
    /// Number of iterations to convergence
    pub iterations: usize,
    
    /// Final energy value
    pub final_energy: f32,
    
    /// Whether Jacobians are used
    pub has_jacobians: bool,
    
    /// Whether residuals are stored
    pub has_residuals: bool,
    
    /// Quantization mode (f16, int8, etc)
    pub quantization: QuantizationMode,
}

/// Quantization modes for storage
#[derive(Debug, Clone, Copy)]
pub enum QuantizationMode {
    F16,
    Int8,
    Int4,
}

/// Statistics for robust reduction (per anchor)
#[repr(C, align(16))]
pub struct AnchorStats {
    /// Robust mean of assigned points [d]
    pub mean: Vec<f32>,
    
    /// Number of points assigned
    pub count: usize,
    
    /// Variance for each dimension
    pub variance: Vec<f32>,
}

impl AnchorStats {
    pub fn new(d: usize) -> Self {
        Self {
            mean: vec![0.0; d],
            count: 0,
            variance: vec![0.0; d],
        }
    }
}

/// Annealing state during optimization
#[derive(Debug, Clone)]
pub struct AnnealState {
    /// Current temperature
    pub temperature: f32,
    
    /// Current iteration
    pub iteration: usize,
    
    /// Current total energy
    pub energy: f32,
    
    /// Assignment changes in last iteration
    pub assignment_changes: usize,
    
    /// Convergence flag
    pub converged: bool,
}

impl AnnealState {
    pub fn new(initial_temp: f32) -> Self {
        Self {
            temperature: initial_temp,
            iteration: 0,
            energy: f32::INFINITY,
            assignment_changes: usize::MAX,
            converged: false,
        }
    }
    
    /// Cool the temperature according to schedule
    pub fn cool(&mut self, cooling_rate: f32) {
        self.temperature *= 1.0 - cooling_rate;
        self.iteration += 1;
    }
    
    /// Check convergence criteria
    pub fn check_convergence(&mut self, energy_tol: f32, min_changes: usize) {
        let energy_stable = (self.energy - energy_tol).abs() < 1e-6;
        let assignments_stable = self.assignment_changes < min_changes;
        self.converged = energy_stable && assignments_stable;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_anchor_indexing() {
        let mut anchors = AnchorSet::new(10, 128);
        assert_eq!(anchors.anchors.len(), 10 * 128);
        
        // Test get/set
        anchors.get_anchor_mut(0)[0] = f16::from_f32(1.0);
        assert_eq!(anchors.get_anchor(0)[0], f16::from_f32(1.0));
    }
    
    #[test]
    fn test_assignment_counting() {
        let mut assignments = Assignments::new(100);
        assignments.assign[0] = 1;
        assignments.assign[1] = 1;
        assignments.assign[2] = 2;
        
        let counts = assignments.count_per_anchor(5);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 1);
    }
}