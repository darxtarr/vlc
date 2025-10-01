//! Annealing loop for VLC optimization

use crate::types::{AnchorSet, Assignments, CompressedIndex, IndexMetadata, AnnealState};
use crate::ops;

#[cfg(feature = "gpu-wgpu")]
use crate::gpu::{GpuContext, GpuOps};

#[cfg(feature = "gpu-cuda")]
use crate::gpu::{CudaContext, CudaOps};

use half::f16;

/// Configuration for annealing process
#[derive(Debug, Clone)]
pub struct AnnealingConfig {
    /// Number of anchors
    pub m: usize,
    
    /// Initial temperature
    pub initial_temp: f32,
    
    /// Cooling rate per iteration
    pub cooling_rate: f32,
    
    /// Learning rate for anchor updates
    pub learning_rate: f32,
    
    /// Trimming percentage for robust mean (0.0 to 0.5)
    pub trim_percent: f32,
    
    /// Maximum iterations
    pub max_iterations: usize,
    
    /// Energy tolerance for convergence
    pub energy_tolerance: f32,
    
    /// Minimum assignment changes for convergence
    pub min_assignment_changes: usize,
    
    /// How often to run maintenance (merge/split)
    pub maintenance_interval: usize,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            m: 1024,
            initial_temp: 1.0,
            cooling_rate: 0.01,
            learning_rate: 0.1,
            trim_percent: 0.1,
            max_iterations: 1000,
            energy_tolerance: 1e-4,
            min_assignment_changes: 10,
            maintenance_interval: 50,
        }
    }
}

/// Main compression function
pub fn compress(
    points: &[f16],
    n: usize,
    d: usize,
    config: AnnealingConfig,
) -> CompressedIndex {
    // Initialize anchors (for now, random subset of points)
    let mut anchors = initialize_anchors(points, n, d, config.m);
    
    // Initialize state
    let mut state = AnnealState::new(config.initial_temp);
    let mut assignments = Assignments::new(n);
    let mut prev_energy = f32::INFINITY;
    let mut stable_iters = 0; // Track consecutive stable iterations

    // Main annealing loop
    while state.iteration < config.max_iterations && !state.converged {
        // Store old assignments
        let old_assignments = assignments.clone();
        
        // Step 1: Assign points to anchors
        assignments = ops::assign_points(points, &anchors, n, d);
        
        // Step 2: Compute robust statistics
        let stats = ops::compute_robust_stats(
            points,
            &assignments,
            &anchors,
            config.trim_percent,
        );
        
        // Step 3: Update anchors
        ops::update_anchors(
            &mut anchors,
            &stats,
            state.temperature,
            config.learning_rate,
        );
        
        // Step 4: Maintenance (if interval reached)
        if state.iteration > 0 && state.iteration % config.maintenance_interval == 0 {
            // Merge close anchors (threshold: 10% of average anchor distance)
            let avg_anchor_dist = estimate_average_anchor_distance(&anchors);
            let merge_threshold = avg_anchor_dist * 0.1;
            let merged = ops::merge_close_anchors(&mut anchors, &mut assignments, merge_threshold, d);

            // Split overloaded anchors (threshold: 2x expected count)
            let expected_count = n / anchors.m;
            let split_threshold = expected_count * 2;
            let split = ops::split_overloaded_anchors(
                &mut anchors,
                &mut assignments,
                points,
                n,
                d,
                split_threshold,
                state.temperature * 2.0, // variance threshold scales with temperature
            );

            if merged > 0 || split > 0 {
                println!("  Maintenance: merged={}, split={}, anchors={}",
                         merged, split, anchors.m);
            }
        }
        
        // Compute energy and changes
        state.energy = ops::compute_energy(points, &anchors, &assignments);
        state.assignment_changes = ops::count_assignment_changes(&old_assignments, &assignments);

        // Check convergence: both energy stable AND assignments stable
        let energy_change = (state.energy - prev_energy).abs();
        let energy_stable = energy_change < config.energy_tolerance;
        let assignments_stable = state.assignment_changes < config.min_assignment_changes;

        if energy_stable && assignments_stable {
            stable_iters += 1;
            // Require 3 consecutive stable iterations to declare convergence
            if stable_iters >= 3 {
                state.converged = true;
                println!("Converged after {} iterations (stable for {} iters)",
                         state.iteration, stable_iters);
            }
        } else {
            stable_iters = 0; // Reset if not stable
        }

        // Cool temperature
        state.cool(config.cooling_rate);

        // Log progress (minimal)
        if state.iteration % 10 == 0 {
            println!(
                "Iter {}: T={:.4}, E={:.4}, ΔE={:.6}, Changes={}",
                state.iteration, state.temperature, state.energy, energy_change, state.assignment_changes
            );
        }

        prev_energy = state.energy;
    }
    
    // Create compressed index
    let metadata = IndexMetadata {
        n_original: n,
        d_original: d,
        compression_ratio: compute_compression_ratio(n, d, config.m, &assignments),
        iterations: state.iteration,
        final_energy: state.energy,
        has_jacobians: false,
        has_residuals: false,
        quantization: crate::types::QuantizationMode::F16,
    };
    
    CompressedIndex {
        anchor_set: anchors,
        assignments,
        metadata,
    }
}

/// Initialize anchors using k-means++ for better spread
fn initialize_anchors(points: &[f16], n: usize, d: usize, m: usize) -> AnchorSet {
    let mut anchors = AnchorSet::new(m, d);

    // k-means++ initialization
    // 1. Pick first anchor randomly (use point 0 for determinism)
    let first_point = 0;
    let anchor = anchors.get_anchor_mut(0);
    for dim in 0..d {
        anchor[dim] = points[first_point * d + dim];
    }

    // 2. For each subsequent anchor, pick proportional to D(x)²
    let mut distances = vec![f32::INFINITY; n];

    for anchor_idx in 1..m {
        // Compute D(x)² = distance to nearest existing anchor
        for point_idx in 0..n {
            let point_start = point_idx * d;
            let point_slice = &points[point_start..point_start + d];

            // Find min distance to any existing anchor
            let mut min_dist = f32::INFINITY;
            for prev_anchor_idx in 0..anchor_idx {
                let prev_anchor = anchors.get_anchor(prev_anchor_idx);
                let dist = l2_distance_f16(point_slice, prev_anchor);
                min_dist = min_dist.min(dist);
            }
            distances[point_idx] = min_dist;
        }

        // Pick next anchor proportional to D(x)²
        // For determinism without rand, pick the point with max distance
        let (next_idx, _) = distances.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Copy selected point as next anchor
        let anchor = anchors.get_anchor_mut(anchor_idx);
        let point_start = next_idx * d;
        for dim in 0..d {
            anchor[dim] = points[point_start + dim];
        }
    }

    anchors
}

/// Compute L2 distance between two f16 vectors (helper for initialization)
fn l2_distance_f16(a: &[f16], b: &[f16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i].to_f32() - b[i].to_f32();
        sum += diff * diff;
    }
    sum
}

/// GPU-accelerated compression function (WGPU backend)
///
/// This is a parallel implementation that uses GPU for expensive operations
/// while keeping CPU compress() completely untouched for safety.
#[cfg(feature = "gpu-wgpu")]
pub async fn compress_gpu(
    points: &[f16],
    n: usize,
    d: usize,
    config: AnnealingConfig,
) -> Result<CompressedIndex, Box<dyn std::error::Error>> {
    // Initialize GPU context
    let gpu_ctx = std::sync::Arc::new(GpuContext::new().await?);
    let mut gpu_ops = GpuOps::new(gpu_ctx);

    // Initialize anchors (same k-means++ as CPU)
    let mut anchors = initialize_anchors(points, n, d, config.m);

    // Initialize state
    let mut state = AnnealState::new(config.initial_temp);
    let mut assignments = Assignments::new(n);
    let mut prev_energy = f32::INFINITY;
    let mut stable_iters = 0;

    println!("GPU compression: n={}, d={}, m={}", n, d, config.m);

    // Main annealing loop
    while state.iteration < config.max_iterations && !state.converged {
        // Store old assignments
        let old_assignments = assignments.clone();

        // Step 1: Assign points to anchors (GPU-accelerated!)
        assignments = gpu_ops.assign_points(points, &anchors, n, d).await?;

        // Step 2: Compute robust statistics (GPU)
        let stats = gpu_ops.reduce_stats(
            points,
            &assignments,
            &anchors,
            n,
            anchors.m,
            d,
        ).await?;

        // Step 3: Update anchors (GPU)
        gpu_ops.update_anchors(
            &mut anchors,
            &stats,
            state.temperature,
            config.learning_rate,
        ).await?;

        // Step 4: Maintenance (if interval reached)
        if state.iteration > 0 && state.iteration % config.maintenance_interval == 0 {
            // Merge close anchors (threshold: 10% of average anchor distance)
            let avg_anchor_dist = estimate_average_anchor_distance(&anchors);
            let merge_threshold = avg_anchor_dist * 0.1;
            let merged = ops::merge_close_anchors(&mut anchors, &mut assignments, merge_threshold, d);

            // Split overloaded anchors (threshold: 2x expected count)
            let expected_count = n / anchors.m;
            let split_threshold = expected_count * 2;
            let split = ops::split_overloaded_anchors(
                &mut anchors,
                &mut assignments,
                points,
                n,
                d,
                split_threshold,
                state.temperature * 2.0, // variance threshold scales with temperature
            );

            if merged > 0 || split > 0 {
                println!("  Maintenance: merged={}, split={}, anchors={}",
                         merged, split, anchors.m);
            }
        }

        // Compute energy and changes (CPU)
        state.energy = ops::compute_energy(points, &anchors, &assignments);
        state.assignment_changes = ops::count_assignment_changes(&old_assignments, &assignments);

        // Check convergence: both energy stable AND assignments stable
        let energy_change = (state.energy - prev_energy).abs();
        let energy_stable = energy_change < config.energy_tolerance;
        let assignments_stable = state.assignment_changes < config.min_assignment_changes;

        if energy_stable && assignments_stable {
            stable_iters += 1;
            // Require 3 consecutive stable iterations to declare convergence
            if stable_iters >= 3 {
                state.converged = true;
                println!("GPU: Converged after {} iterations (stable for {} iters)",
                         state.iteration, stable_iters);
            }
        } else {
            stable_iters = 0; // Reset if not stable
        }

        // Cool temperature
        state.cool(config.cooling_rate);

        // Log progress (minimal)
        if state.iteration % 10 == 0 {
            println!(
                "GPU Iter {}: T={:.4}, E={:.4}, ΔE={:.6}, Changes={}",
                state.iteration, state.temperature, state.energy, energy_change, state.assignment_changes
            );
        }

        prev_energy = state.energy;
    }

    // Create compressed index
    let metadata = IndexMetadata {
        n_original: n,
        d_original: d,
        compression_ratio: compute_compression_ratio(n, d, config.m, &assignments),
        iterations: state.iteration,
        final_energy: state.energy,
        has_jacobians: false,
        has_residuals: false,
        quantization: crate::types::QuantizationMode::F16,
    };

    Ok(CompressedIndex {
        anchor_set: anchors,
        assignments,
        metadata,
    })
}

/// CUDA-accelerated compression function
///
/// Similar to compress_gpu but using CUDA backend for NVIDIA GPUs.
/// This is synchronous unlike the WGPU version.
#[cfg(feature = "gpu-cuda")]
pub fn compress_cuda(
    points: &[f16],
    n: usize,
    d: usize,
    config: AnnealingConfig,
) -> Result<CompressedIndex, Box<dyn std::error::Error>> {
    // Initialize CUDA context
    let cuda_ctx = std::sync::Arc::new(CudaContext::new(0)?);
    let mut cuda_ops = CudaOps::new(cuda_ctx);

    // Initialize anchors (same k-means++ as CPU)
    let mut anchors = initialize_anchors(points, n, d, config.m);

    // Initialize state
    let mut state = AnnealState::new(config.initial_temp);
    let mut assignments = Assignments::new(n);
    let mut prev_energy = f32::INFINITY;
    let mut stable_iters = 0;

    println!("CUDA compression: n={}, d={}, m={}", n, d, config.m);

    // Main annealing loop
    while state.iteration < config.max_iterations && !state.converged {
        // Store old assignments
        let old_assignments = assignments.clone();

        // Step 1: Assign points to anchors (CUDA-accelerated!)
        assignments = cuda_ops.assign_points(points, &anchors, n, d)?;

        // Step 2: Compute robust statistics (CUDA)
        let stats = cuda_ops.reduce_stats(
            points,
            &assignments,
            &anchors,
            n,
            anchors.m,
            d,
        )?;

        // Step 3: Update anchors (CUDA)
        cuda_ops.update_anchors(
            &mut anchors,
            &stats,
            state.temperature,
            config.learning_rate,
        )?;

        // Step 4: Maintenance (if interval reached)
        if state.iteration > 0 && state.iteration % config.maintenance_interval == 0 {
            // Merge close anchors (threshold: 10% of average anchor distance)
            let avg_anchor_dist = estimate_average_anchor_distance(&anchors);
            let merge_threshold = avg_anchor_dist * 0.1;
            let merged = ops::merge_close_anchors(&mut anchors, &mut assignments, merge_threshold, d);

            // Split overloaded anchors (threshold: 2x expected count)
            let expected_count = n / anchors.m;
            let split_threshold = expected_count * 2;
            let split = ops::split_overloaded_anchors(
                &mut anchors,
                &mut assignments,
                points,
                n,
                d,
                split_threshold,
                state.temperature * 2.0, // variance threshold scales with temperature
            );

            if merged > 0 || split > 0 {
                println!("  Maintenance: merged={}, split={}, anchors={}",
                         merged, split, anchors.m);
            }
        }

        // Compute energy and changes (CPU)
        state.energy = ops::compute_energy(points, &anchors, &assignments);
        state.assignment_changes = ops::count_assignment_changes(&old_assignments, &assignments);

        // Check convergence: both energy stable AND assignments stable
        let energy_change = (state.energy - prev_energy).abs();
        let energy_stable = energy_change < config.energy_tolerance;
        let assignments_stable = state.assignment_changes < config.min_assignment_changes;

        if energy_stable && assignments_stable {
            stable_iters += 1;
            // Require 3 consecutive stable iterations to declare convergence
            if stable_iters >= 3 {
                state.converged = true;
                println!("CUDA: Converged after {} iterations (stable for {} iters)",
                         state.iteration, stable_iters);
            }
        } else {
            stable_iters = 0; // Reset if not stable
        }

        // Cool temperature
        state.cool(config.cooling_rate);

        // Log progress (minimal)
        if state.iteration % 10 == 0 {
            println!(
                "CUDA Iter {}: T={:.4}, E={:.4}, delta_E={:.6}, Changes={}",
                state.iteration, state.temperature, state.energy, energy_change, state.assignment_changes
            );
        }

        prev_energy = state.energy;
    }

    // Create compressed index
    let metadata = IndexMetadata {
        n_original: n,
        d_original: d,
        compression_ratio: compute_compression_ratio(n, d, config.m, &assignments),
        iterations: state.iteration,
        final_energy: state.energy,
        has_jacobians: false,
        has_residuals: false,
        quantization: crate::types::QuantizationMode::F16,
    };

    Ok(CompressedIndex {
        anchor_set: anchors,
        assignments,
        metadata,
    })
}

/// Compute compression ratio
fn compute_compression_ratio(
    n: usize,
    d: usize,
    m: usize,
    assignments: &Assignments,
) -> f32 {
    let original_bytes = n * d * 4; // Assuming f32 originals
    let compressed_bytes =
        m * d * 2 + // anchors as f16
        n * 4 +     // assignments as u32
        if assignments.residuals.is_some() {
            n * assignments.d_r * 2 // residuals as f16
        } else {
            0
        };

    compressed_bytes as f32 / original_bytes as f32
}

/// Estimate average distance between anchors (for merge threshold)
fn estimate_average_anchor_distance(anchors: &AnchorSet) -> f32 {
    if anchors.m < 2 {
        return 1.0; // Default if too few anchors
    }

    let mut total_dist = 0.0f32;
    let mut count = 0usize;

    // Sample a subset of anchor pairs to estimate average distance
    let sample_size = (anchors.m * 2).min(100); // Sample up to 100 pairs

    for i in 0..sample_size.min(anchors.m) {
        let j = (i + anchors.m / 2) % anchors.m; // Sample distant pairs
        let anchor_i = anchors.get_anchor(i);
        let anchor_j = anchors.get_anchor(j);

        total_dist += l2_distance_f16(anchor_i, anchor_j).sqrt();
        count += 1;
    }

    if count > 0 {
        total_dist / count as f32
    } else {
        1.0
    }
}