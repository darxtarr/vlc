//! Maintenance operations for anchor management
//!
//! These operations keep the anchor set healthy during and after compression:
//! - Merge: Combine anchors that are too close together
//! - Split: Divide anchors that are overloaded with too many points
//!
//! These operations help maintain a good balance between compression ratio
//! and reconstruction quality.

use half::f16;
use crate::types::{AnchorSet, Assignments};

/// Result of a maintenance operation
#[derive(Debug)]
pub struct MaintenanceResult {
    pub anchors_merged: usize,
    pub anchors_split: usize,
    pub final_anchor_count: usize,
}

impl MaintenanceResult {
    pub fn new() -> Self {
        Self {
            anchors_merged: 0,
            anchors_split: 0,
            final_anchor_count: 0,
        }
    }
}

/// Merge anchors that are closer than the threshold distance
///
/// When two anchors are very close, they're redundant. We merge them into
/// a single anchor positioned at their weighted centroid (weighted by point count).
/// All points assigned to either anchor are reassigned to the merged anchor.
///
/// # Arguments
/// * `anchors` - The anchor set (will be modified in place)
/// * `assignments` - Point assignments (will be updated)
/// * `merge_threshold` - Maximum distance for merging (typically 0.05-0.2)
/// * `d` - Dimensionality
///
/// # Returns
/// Number of anchors that were merged (removed)
pub fn merge_close_anchors(
    anchors: &mut AnchorSet,
    assignments: &mut Assignments,
    merge_threshold: f32,
    d: usize,
) -> usize {
    let m = anchors.m;
    let threshold_sq = merge_threshold * merge_threshold;

    // Track which anchors should be merged (anchor_idx -> merge_into_idx)
    let mut merge_map: Vec<Option<usize>> = vec![None; m];
    let anchor_counts = assignments.count_per_anchor(m);

    // Find pairs of anchors to merge
    for i in 0..m {
        if merge_map[i].is_some() {
            continue; // Already marked for merging
        }

        let anchor_i = anchors.get_anchor(i);

        for j in (i + 1)..m {
            if merge_map[j].is_some() {
                continue; // Already marked for merging
            }

            let anchor_j = anchors.get_anchor(j);

            // Compute squared distance between anchors
            let dist_sq = l2_distance_sq_f16(anchor_i, anchor_j);

            if dist_sq < threshold_sq {
                // Mark j for merging into i
                merge_map[j] = Some(i);
            }
        }
    }

    // Count how many anchors are being merged
    let merged_count = merge_map.iter().filter(|x| x.is_some()).count();

    if merged_count == 0 {
        return 0;
    }

    // Create merged anchor positions (weighted by point counts)
    let mut new_positions: Vec<Vec<f32>> = vec![vec![0.0; d]; m];
    let mut new_counts = vec![0usize; m];

    for anchor_idx in 0..m {
        let target_idx = merge_map[anchor_idx].unwrap_or(anchor_idx);
        let count = anchor_counts[anchor_idx];

        if count == 0 {
            continue;
        }

        let anchor = anchors.get_anchor(anchor_idx);

        // Add weighted contribution to target anchor
        for dim in 0..d {
            new_positions[target_idx][dim] += anchor[dim].to_f32() * count as f32;
        }
        new_counts[target_idx] += count;
    }

    // Normalize by counts to get weighted centroids
    for anchor_idx in 0..m {
        if new_counts[anchor_idx] > 0 {
            let count = new_counts[anchor_idx] as f32;
            for dim in 0..d {
                new_positions[anchor_idx][dim] /= count;
            }
        }
    }

    // Update anchor positions for merged anchors
    for anchor_idx in 0..m {
        if new_counts[anchor_idx] > 0 && merge_map[anchor_idx].is_none() {
            // This anchor is a merge target or standalone
            let anchor_mut = anchors.get_anchor_mut(anchor_idx);
            for dim in 0..d {
                anchor_mut[dim] = f16::from_f32(new_positions[anchor_idx][dim]);
            }
        }
    }

    // Update assignments to point to merge targets
    for assign in assignments.assign.iter_mut() {
        if let Some(target_idx) = merge_map[*assign as usize] {
            *assign = target_idx as u32;
        }
    }

    merged_count
}

/// Split anchors that have too many points assigned or high variance
///
/// When an anchor is "overloaded" with too many points, or the variance
/// of assigned points is too high, we split it into two new anchors.
/// The split uses a simple 2-means clustering approach.
///
/// # Arguments
/// * `anchors` - The anchor set (will be modified/grown)
/// * `assignments` - Point assignments (will be updated)
/// * `points` - Original point data
/// * `n` - Number of points
/// * `d` - Dimensionality
/// * `split_threshold_count` - Maximum points per anchor before splitting
/// * `split_threshold_var` - Maximum variance before splitting
///
/// # Returns
/// Number of anchors that were split (new anchors created)
pub fn split_overloaded_anchors(
    anchors: &mut AnchorSet,
    assignments: &mut Assignments,
    points: &[f16],
    n: usize,
    d: usize,
    split_threshold_count: usize,
    split_threshold_var: f32,
) -> usize {
    let m = anchors.m;
    let mut splits_performed = 0;

    // Compute current statistics
    let anchor_counts = assignments.count_per_anchor(m);
    let anchor_variances = compute_anchor_variances(anchors, assignments, points, n, d);

    // Find anchors that need splitting
    let mut anchors_to_split = Vec::new();
    for anchor_idx in 0..m {
        let count = anchor_counts[anchor_idx];
        let max_var = anchor_variances[anchor_idx];

        if count > split_threshold_count || max_var > split_threshold_var {
            anchors_to_split.push(anchor_idx);
        }
    }

    if anchors_to_split.is_empty() {
        return 0;
    }

    // Perform splits (we'll add new anchors at the end)
    let mut new_anchor_positions = Vec::new();
    let mut reassignments = Vec::new(); // (point_idx, new_anchor_idx)

    for &anchor_idx in &anchors_to_split {
        // Collect points assigned to this anchor
        let mut assigned_point_indices = Vec::new();
        for (point_idx, &assign) in assignments.assign.iter().enumerate() {
            if assign == anchor_idx as u32 {
                assigned_point_indices.push(point_idx);
            }
        }

        if assigned_point_indices.len() < 2 {
            continue; // Can't split less than 2 points
        }

        // Perform 2-means split
        let (pos1, pos2, split_assignments) = split_two_means(
            points,
            &assigned_point_indices,
            d,
        );

        // First new anchor replaces the old one
        let anchor_mut = anchors.get_anchor_mut(anchor_idx);
        for dim in 0..d {
            anchor_mut[dim] = f16::from_f32(pos1[dim]);
        }

        // Second new anchor will be added to the end
        let new_anchor_idx = m + new_anchor_positions.len();
        new_anchor_positions.push(pos2);

        // Record reassignments
        for (i, &point_idx) in assigned_point_indices.iter().enumerate() {
            if split_assignments[i] == 1 {
                // Assign to new anchor
                reassignments.push((point_idx, new_anchor_idx));
            }
        }

        splits_performed += 1;
    }

    // Add new anchors to the anchor set
    for new_pos in new_anchor_positions {
        anchors.anchors.reserve(d);
        for val in new_pos {
            anchors.anchors.push(f16::from_f32(val));
        }
        anchors.m += 1;
    }

    // Apply reassignments
    for (point_idx, new_anchor_idx) in reassignments {
        assignments.assign[point_idx] = new_anchor_idx as u32;
    }

    splits_performed
}

/// Compute variance of points assigned to each anchor
fn compute_anchor_variances(
    anchors: &AnchorSet,
    assignments: &Assignments,
    points: &[f16],
    n: usize,
    d: usize,
) -> Vec<f32> {
    let m = anchors.m;
    let mut variances = vec![0.0f32; m];

    for anchor_idx in 0..m {
        let anchor = anchors.get_anchor(anchor_idx);
        let mut sum_sq = 0.0f32;
        let mut count = 0usize;

        for point_idx in 0..n {
            if assignments.assign[point_idx] == anchor_idx as u32 {
                let point_start = point_idx * d;
                let point = &points[point_start..point_start + d];

                let dist_sq = l2_distance_sq_f16(point, anchor);
                sum_sq += dist_sq;
                count += 1;
            }
        }

        if count > 0 {
            variances[anchor_idx] = sum_sq / count as f32;
        }
    }

    variances
}

/// Perform 2-means clustering on a set of points
///
/// Returns (centroid1, centroid2, assignments)
fn split_two_means(
    points: &[f16],
    point_indices: &[usize],
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let n_points = point_indices.len();

    // Initialize with first and last point as centroids
    let first_idx = point_indices[0];
    let last_idx = point_indices[n_points - 1];

    let mut c1 = points[first_idx * d..(first_idx + 1) * d]
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<f32>>();
    let mut c2 = points[last_idx * d..(last_idx + 1) * d]
        .iter()
        .map(|x| x.to_f32())
        .collect::<Vec<f32>>();

    let mut assignments = vec![0u8; n_points];

    // Run 5 iterations of k-means
    for _iter in 0..5 {
        // Assign to nearest centroid
        for (i, &point_idx) in point_indices.iter().enumerate() {
            let point_start = point_idx * d;
            let point = &points[point_start..point_start + d];

            let dist1 = l2_distance_sq_f32_f16(point, &c1);
            let dist2 = l2_distance_sq_f32_f16(point, &c2);

            assignments[i] = if dist1 < dist2 { 0 } else { 1 };
        }

        // Recompute centroids
        let mut sum1 = vec![0.0f32; d];
        let mut sum2 = vec![0.0f32; d];
        let mut count1 = 0usize;
        let mut count2 = 0usize;

        for (i, &point_idx) in point_indices.iter().enumerate() {
            let point_start = point_idx * d;
            let point = &points[point_start..point_start + d];

            if assignments[i] == 0 {
                for dim in 0..d {
                    sum1[dim] += point[dim].to_f32();
                }
                count1 += 1;
            } else {
                for dim in 0..d {
                    sum2[dim] += point[dim].to_f32();
                }
                count2 += 1;
            }
        }

        // Update centroids
        if count1 > 0 {
            for dim in 0..d {
                c1[dim] = sum1[dim] / count1 as f32;
            }
        }
        if count2 > 0 {
            for dim in 0..d {
                c2[dim] = sum2[dim] / count2 as f32;
            }
        }
    }

    (c1, c2, assignments)
}

/// Compute squared L2 distance between two f16 vectors
fn l2_distance_sq_f16(a: &[f16], b: &[f16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i].to_f32() - b[i].to_f32();
        sum += diff * diff;
    }
    sum
}

/// Compute squared L2 distance between f16 vector and f32 vector
fn l2_distance_sq_f32_f16(a: &[f16], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i].to_f32() - b[i];
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_identical_anchors() {
        let mut anchors = AnchorSet::new(3, 4);

        // Create two identical anchors at position 0 and 1
        for dim in 0..4 {
            anchors.get_anchor_mut(0)[dim] = f16::from_f32(1.0);
            anchors.get_anchor_mut(1)[dim] = f16::from_f32(1.0);
            anchors.get_anchor_mut(2)[dim] = f16::from_f32(5.0);
        }

        // Create assignments
        let mut assignments = Assignments::new(6);
        assignments.assign = vec![0, 0, 1, 1, 2, 2];

        // Merge with high threshold
        let merged = merge_close_anchors(&mut anchors, &mut assignments, 0.5, 4);

        assert_eq!(merged, 1, "Should merge one anchor");

        // All points from anchor 1 should now point to anchor 0
        for i in 0..4 {
            assert!(assignments.assign[i] == 0, "Points should be reassigned to anchor 0");
        }
    }

    #[test]
    fn test_split_two_means() {
        // Create 4 points: 2 at (0,0) and 2 at (10,10)
        let mut points = Vec::new();
        for _ in 0..2 {
            points.push(f16::from_f32(0.0));
            points.push(f16::from_f32(0.0));
        }
        for _ in 0..2 {
            points.push(f16::from_f32(10.0));
            points.push(f16::from_f32(10.0));
        }

        let point_indices = vec![0, 1, 2, 3];
        let (c1, c2, assignments) = split_two_means(&points, &point_indices, 2);

        // Check that centroids are separated
        let dist = ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2)).sqrt();
        assert!(dist > 5.0, "Centroids should be well separated");

        // Check that assignments are sensible (2 points each)
        let count0 = assignments.iter().filter(|&&x| x == 0).count();
        let count1 = assignments.iter().filter(|&&x| x == 1).count();
        assert!(count0 >= 1 && count1 >= 1, "Both clusters should have points");
    }
}
