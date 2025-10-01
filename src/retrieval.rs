//! Compressed retrieval operations
//!
//! This module implements efficient search over compressed embeddings.
//! The strategy is:
//! 1. Find K nearest anchors to the query (cheap - only m comparisons)
//! 2. Gather candidate points assigned to those anchors
//! 3. Reconstruct candidates (anchor + optional residual/jacobian)
//! 4. Compute exact distances to query
//! 5. Return top-k results

use half::f16;
use crate::types::CompressedIndex;

/// Query result: (point_index, distance)
pub type QueryResult = (usize, f32);

impl CompressedIndex {
    /// Query for k nearest neighbors in the compressed space
    ///
    /// # Arguments
    /// * `query` - Query vector (f16, dimension d)
    /// * `k` - Number of neighbors to return
    /// * `num_anchor_candidates` - How many anchors to search (default: k * 2)
    ///
    /// # Returns
    /// Vector of (point_index, distance) tuples, sorted by distance
    pub fn query(&self, query: &[f16], k: usize, num_anchor_candidates: Option<usize>) -> Vec<QueryResult> {
        let _d = self.anchor_set.d;
        let _n = self.assignments.n;

        // Step 1: Find nearest anchors to query
        let num_candidates = num_anchor_candidates.unwrap_or(k * 2).min(self.anchor_set.m);
        let nearest_anchors = self.find_nearest_anchors(query, num_candidates);

        // Step 2: Gather candidate points from those anchors
        let mut candidates = Vec::new();
        for &anchor_idx in &nearest_anchors {
            for (point_idx, &assigned_anchor) in self.assignments.assign.iter().enumerate() {
                if assigned_anchor == anchor_idx as u32 {
                    candidates.push(point_idx);
                }
            }
        }

        // Early return if not enough candidates
        if candidates.is_empty() {
            return Vec::new();
        }

        // Step 3: Compute distances to query (using reconstructed points)
        let mut distances: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&point_idx| {
                let reconstructed = self.reconstruct_point(point_idx);
                let dist = l2_distance_f16(query, &reconstructed);
                (point_idx, dist)
            })
            .collect();

        // Step 4: Sort by distance and return top-k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        distances
    }

    /// Find the nearest K anchors to a query vector
    fn find_nearest_anchors(&self, query: &[f16], k: usize) -> Vec<usize> {
        let m = self.anchor_set.m;
        let k_actual = k.min(m);

        // Compute distances to all anchors
        let mut anchor_distances: Vec<(usize, f32)> = (0..m)
            .map(|anchor_idx| {
                let anchor = self.anchor_set.get_anchor(anchor_idx);
                let dist = l2_distance_f16(query, anchor);
                (anchor_idx, dist)
            })
            .collect();

        // Sort and take top-k
        anchor_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        anchor_distances
            .into_iter()
            .take(k_actual)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Reconstruct a point from its compressed representation
    ///
    /// Currently just returns the assigned anchor position.
    /// TODO: Add residual/jacobian reconstruction when implemented.
    pub fn reconstruct_point(&self, point_idx: usize) -> Vec<f16> {
        let anchor_idx = self.assignments.assign[point_idx] as usize;
        let anchor = self.anchor_set.get_anchor(anchor_idx);

        // For now, just return anchor
        // Later: anchor + residual or anchor + jacobian * delta
        anchor.to_vec()
    }

    /// Batch query for multiple queries at once
    ///
    /// More efficient than calling query() repeatedly.
    pub fn query_batch(
        &self,
        queries: &[f16],
        num_queries: usize,
        k: usize,
        num_anchor_candidates: Option<usize>,
    ) -> Vec<Vec<QueryResult>> {
        let d = self.anchor_set.d;

        (0..num_queries)
            .map(|i| {
                let query_start = i * d;
                let query = &queries[query_start..query_start + d];
                self.query(query, k, num_anchor_candidates)
            })
            .collect()
    }

    /// Compute recall@k against ground truth
    ///
    /// # Arguments
    /// * `queries` - Query vectors
    /// * `num_queries` - Number of queries
    /// * `ground_truth` - Ground truth neighbors (outer vec: queries, inner vec: point indices)
    /// * `k` - Number of neighbors to evaluate
    ///
    /// # Returns
    /// Average recall@k across all queries
    pub fn evaluate_recall(
        &self,
        queries: &[f16],
        num_queries: usize,
        ground_truth: &[Vec<usize>],
        k: usize,
    ) -> f32 {
        let d = self.anchor_set.d;
        let mut total_recall = 0.0;

        for i in 0..num_queries {
            let query_start = i * d;
            let query = &queries[query_start..query_start + d];

            // Get our results
            let results = self.query(query, k, None);
            let result_ids: std::collections::HashSet<usize> =
                results.iter().map(|(idx, _)| *idx).collect();

            // Compare with ground truth
            let gt_ids: std::collections::HashSet<usize> =
                ground_truth[i].iter().take(k).copied().collect();

            let intersection = result_ids.intersection(&gt_ids).count();
            let recall = intersection as f32 / k as f32;
            total_recall += recall;
        }

        total_recall / num_queries as f32
    }
}

/// Compute L2 distance between two f16 vectors
fn l2_distance_f16(a: &[f16], b: &[f16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i].to_f32() - b[i].to_f32();
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnchorSet, Assignments, IndexMetadata, QuantizationMode};

    #[test]
    fn test_query_basic() {
        // Create a simple compressed index
        let d = 4;
        let m = 3;
        let n = 9;

        let mut anchors = AnchorSet::new(m, d);

        // Anchor 0 at origin
        for dim in 0..d {
            anchors.get_anchor_mut(0)[dim] = f16::from_f32(0.0);
        }
        // Anchor 1 at (1, 1, 1, 1)
        for dim in 0..d {
            anchors.get_anchor_mut(1)[dim] = f16::from_f32(1.0);
        }
        // Anchor 2 at (5, 5, 5, 5)
        for dim in 0..d {
            anchors.get_anchor_mut(2)[dim] = f16::from_f32(5.0);
        }

        // Assign 3 points to each anchor
        let mut assignments = Assignments::new(n);
        assignments.assign = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let metadata = IndexMetadata {
            n_original: n,
            d_original: d,
            compression_ratio: 0.1,
            iterations: 10,
            final_energy: 0.5,
            has_jacobians: false,
            has_residuals: false,
            quantization: QuantizationMode::F16,
        };

        let index = CompressedIndex {
            anchor_set: anchors,
            assignments,
            metadata,
        };

        // Query near anchor 1
        let query = vec![
            f16::from_f32(1.1),
            f16::from_f32(1.1),
            f16::from_f32(1.1),
            f16::from_f32(1.1),
        ];

        let results = index.query(&query, 3, None);

        // Should return points assigned to anchor 1
        assert_eq!(results.len(), 3);
        assert!(results[0].0 >= 3 && results[0].0 <= 5, "Should be from anchor 1");
    }

    #[test]
    fn test_reconstruct_point() {
        let d = 2;
        let m = 2;
        let n = 4;

        let mut anchors = AnchorSet::new(m, d);
        anchors.get_anchor_mut(0)[0] = f16::from_f32(1.0);
        anchors.get_anchor_mut(0)[1] = f16::from_f32(2.0);

        let mut assignments = Assignments::new(n);
        assignments.assign[0] = 0;

        let metadata = IndexMetadata {
            n_original: n,
            d_original: d,
            compression_ratio: 0.1,
            iterations: 10,
            final_energy: 0.5,
            has_jacobians: false,
            has_residuals: false,
            quantization: QuantizationMode::F16,
        };

        let index = CompressedIndex {
            anchor_set: anchors,
            assignments,
            metadata,
        };

        let reconstructed = index.reconstruct_point(0);
        assert_eq!(reconstructed[0], f16::from_f32(1.0));
        assert_eq!(reconstructed[1], f16::from_f32(2.0));
    }
}
