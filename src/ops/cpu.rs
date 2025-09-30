//! CPU implementations of core VLC operations
//! 
//! These serve as reference implementations and fallbacks.
//! All operations work with f32 internally for stability.

use half::f16;
use crate::types::{AnchorSet, Assignments, AnchorStats};

/// Assign each point to its nearest anchor
pub fn assign_points(
    points: &[f16],
    anchors: &AnchorSet,
    n: usize,
    d: usize,
) -> Assignments {
    let mut assignments = Assignments::new(n);
    
    for point_idx in 0..n {
        let point_start = point_idx * d;
        let point = &points[point_start..point_start + d];
        
        let mut min_dist = f32::INFINITY;
        let mut min_anchor = 0u32;
        
        for anchor_idx in 0..anchors.m {
            let anchor = anchors.get_anchor(anchor_idx);
            
            // Compute L2 distance
            let dist = l2_distance_f16(point, anchor);
            
            if dist < min_dist {
                min_dist = dist;
                min_anchor = anchor_idx as u32;
            }
        }
        
        assignments.assign[point_idx] = min_anchor;
    }
    
    assignments
}

/// Compute L2 distance between two f16 vectors
fn l2_distance_f16(a: &[f16], b: &[f16]) -> f32 {
    let mut sum = 0.0f32;
    
    // Process 4 at a time for better CPU vectorization
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx].to_f32() - b[idx].to_f32();
        let d1 = a[idx + 1].to_f32() - b[idx + 1].to_f32();
        let d2 = a[idx + 2].to_f32() - b[idx + 2].to_f32();
        let d3 = a[idx + 3].to_f32() - b[idx + 3].to_f32();
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    
    // Handle remainder
    for i in (chunks * 4)..a.len() {
        let diff = a[i].to_f32() - b[i].to_f32();
        sum += diff * diff;
    }
    
    sum
}

/// Compute robust mean for points assigned to each anchor
pub fn compute_robust_stats(
    points: &[f16],
    assignments: &Assignments,
    anchors: &AnchorSet,
    trim_percent: f32,
) -> Vec<AnchorStats> {
    let mut stats = Vec::with_capacity(anchors.m);
    
    for anchor_idx in 0..anchors.m {
        let mut anchor_stats = AnchorStats::new(anchors.d);
        
        // Collect points assigned to this anchor
        let mut assigned_points = Vec::new();
        for (point_idx, &assign) in assignments.assign.iter().enumerate() {
            if assign == anchor_idx as u32 {
                let start = point_idx * anchors.d;
                assigned_points.push(&points[start..start + anchors.d]);
            }
        }
        
        if assigned_points.is_empty() {
            stats.push(anchor_stats);
            continue;
        }
        
        anchor_stats.count = assigned_points.len();
        
        // Compute distances for trimming
        let anchor = anchors.get_anchor(anchor_idx);
        let mut distances: Vec<(f32, usize)> = assigned_points
            .iter()
            .enumerate()
            .map(|(idx, point)| (l2_distance_f16(point, anchor), idx))
            .collect();
        
        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Determine trimming boundaries
        let trim_count = (assigned_points.len() as f32 * trim_percent) as usize;
        let start_idx = trim_count;
        let end_idx = assigned_points.len().saturating_sub(trim_count);
        
        // Compute trimmed mean
        let mut sum = vec![0.0f32; anchors.d];
        let mut count = 0;
        
        for i in start_idx..end_idx {
            let point_idx = distances[i].1;
            let point = assigned_points[point_idx];
            
            for (dim, &val) in point.iter().enumerate() {
                sum[dim] += val.to_f32();
            }
            count += 1;
        }
        
        // Store mean
        if count > 0 {
            for dim in 0..anchors.d {
                anchor_stats.mean[dim] = sum[dim] / count as f32;
            }
        }
        
        // Compute variance (simplified - using all points, not just trimmed)
        for point in &assigned_points {
            for (dim, &val) in point.iter().enumerate() {
                let diff = val.to_f32() - anchor_stats.mean[dim];
                anchor_stats.variance[dim] += diff * diff;
            }
        }
        
        // Normalize variance
        if anchor_stats.count > 1 {
            for dim in 0..anchors.d {
                anchor_stats.variance[dim] /= (anchor_stats.count - 1) as f32;
            }
        }
        
        stats.push(anchor_stats);
    }
    
    stats
}

/// Update anchors based on statistics and temperature
pub fn update_anchors(
    anchors: &mut AnchorSet,
    stats: &[AnchorStats],
    temperature: f32,
    learning_rate: f32,
) {
    let d = anchors.d; // Cache to avoid borrow issue

    for (anchor_idx, stat) in stats.iter().enumerate() {
        if stat.count == 0 {
            continue; // Skip anchors with no assignments
        }

        // Temperature-scaled learning rate
        let temp_lr = learning_rate * (temperature / (1.0 + temperature));

        let anchor = anchors.get_anchor_mut(anchor_idx);

        for dim in 0..d {
            let old_val = anchor[dim].to_f32();
            let target = stat.mean[dim];

            // Gradient step toward mean
            let step = (target - old_val) * temp_lr;

            // Clamp step size for stability
            let clamped_step = step.max(-0.1).min(0.1);

            // Update
            anchor[dim] = f16::from_f32(old_val + clamped_step);
        }
    }
}

/// Compute total energy (distortion)
pub fn compute_energy(
    points: &[f16],
    anchors: &AnchorSet,
    assignments: &Assignments,
) -> f32 {
    let mut total_distortion = 0.0f32;
    
    for (point_idx, &anchor_idx) in assignments.assign.iter().enumerate() {
        let point_start = point_idx * anchors.d;
        let point = &points[point_start..point_start + anchors.d];
        let anchor = anchors.get_anchor(anchor_idx as usize);
        
        total_distortion += l2_distance_f16(point, anchor);
    }
    
    total_distortion / assignments.n as f32
}

/// Count assignment changes between old and new
pub fn count_assignment_changes(old: &Assignments, new: &Assignments) -> usize {
    old.assign
        .iter()
        .zip(new.assign.iter())
        .filter(|(a, b)| a != b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_distance() {
        let a = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let b = vec![f16::from_f32(4.0), f16::from_f32(5.0), f16::from_f32(6.0)];
        
        let dist = l2_distance_f16(&a, &b);
        let expected = 9.0 + 9.0 + 9.0; // (3)^2 + (3)^2 + (3)^2
        
        assert!((dist - expected).abs() < 1e-5);
    }
    
    #[test]
    fn test_assignment() {
        // Create simple test data
        let points = vec![
            f16::from_f32(0.0), f16::from_f32(0.0),  // Point 0: (0, 0)
            f16::from_f32(10.0), f16::from_f32(10.0), // Point 1: (10, 10)
        ];
        
        let mut anchors = AnchorSet::new(2, 2);
        anchors.get_anchor_mut(0)[0] = f16::from_f32(1.0);
        anchors.get_anchor_mut(0)[1] = f16::from_f32(1.0);  // Anchor 0: (1, 1)
        anchors.get_anchor_mut(1)[0] = f16::from_f32(9.0);
        anchors.get_anchor_mut(1)[1] = f16::from_f32(9.0);  // Anchor 1: (9, 9)
        
        let assignments = assign_points(&points, &anchors, 2, 2);
        
        assert_eq!(assignments.assign[0], 0); // Point 0 closer to Anchor 0
        assert_eq!(assignments.assign[1], 1); // Point 1 closer to Anchor 1
    }
}