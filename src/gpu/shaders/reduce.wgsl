// Robust reduction kernel - computes statistics for points assigned to each anchor
// Based on the spec in docs/KERNELS.md with Huber robust estimation

struct Params {
    n: u32,               // number of points
    m: u32,               // number of anchors
    d: u32,               // dimension
    huber_threshold: f32, // threshold for robust estimation
}

@group(0) @binding(0) var<storage, read> points: array<f32>;        // [n × d]
@group(0) @binding(1) var<storage, read> assigns: array<u32>;       // [n]
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;   // [m × (d+2)]
// stats layout: [mean_0..mean_d-1, count, variance]

@group(1) @binding(0) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// Shared memory for workgroup reduction
var<workgroup> partial_sums: array<f32, 256>;  // reused for each dimension
var<workgroup> total_weight: atomic<u32>;      // total weight (as u32 for atomics)

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let anchor_id = workgroup_id.x;
    let thread_id = local_id.x;

    // Bounds check
    if (anchor_id >= params.m) {
        return;
    }

    // Initialize shared memory
    if (thread_id == 0u) {
        atomicStore(&total_weight, 0u);
    }
    workgroupBarrier();

    // Step 1: Compute robust mean for each dimension
    for (var dim = 0u; dim < params.d; dim++) {
        // Initialize partial sum for this dimension
        partial_sums[thread_id] = 0.0;
        workgroupBarrier();

        // Each thread processes multiple points
        for (var point_id = thread_id; point_id < params.n; point_id += WORKGROUP_SIZE) {
            if (assigns[point_id] == anchor_id) {
                let point_value = points[point_id * params.d + dim];

                // Compute distance from current anchor estimate (simplified for first pass)
                // In practice, this would use the current anchor position
                let weight = 1.0; // Huber weighting will be added in refinement

                partial_sums[thread_id] += point_value * weight;

                // Count points only in first dimension to avoid double counting
                if (dim == 0u) {
                    atomicAdd(&total_weight, 1u);
                }
            }
        }

        workgroupBarrier();

        // Tree reduction of partial sums
        var active_threads = WORKGROUP_SIZE / 2u;
        while (active_threads > 0u) {
            if (thread_id < active_threads) {
                partial_sums[thread_id] += partial_sums[thread_id + active_threads];
            }
            workgroupBarrier();
            active_threads /= 2u;
        }

        // Thread 0 writes the mean for this dimension
        if (thread_id == 0u) {
            let weight_f32 = f32(atomicLoad(&total_weight));
            if (weight_f32 > 0.0) {
                stats[anchor_id * (params.d + 2u) + dim] = partial_sums[0] / weight_f32;
            } else {
                stats[anchor_id * (params.d + 2u) + dim] = 0.0;
            }
        }

        workgroupBarrier();
    }

    // Step 2: Write count and compute simple variance
    if (thread_id == 0u) {
        let weight_f32 = f32(atomicLoad(&total_weight));
        stats[anchor_id * (params.d + 2u) + params.d] = weight_f32;

        // Simple variance computation (can be refined with Huber later)
        var total_variance = 0.0;
        for (var point_id = 0u; point_id < params.n; point_id++) {
            if (assigns[point_id] == anchor_id) {
                var point_variance = 0.0;
                for (var dim = 0u; dim < params.d; dim++) {
                    let point_val = points[point_id * params.d + dim];
                    let mean_val = stats[anchor_id * (params.d + 2u) + dim];
                    let diff = point_val - mean_val;
                    point_variance += diff * diff;
                }
                total_variance += point_variance;
            }
        }

        if (weight_f32 > 1.0) {
            stats[anchor_id * (params.d + 2u) + params.d + 1u] = total_variance / (weight_f32 - 1.0);
        } else {
            stats[anchor_id * (params.d + 2u) + params.d + 1u] = 0.0;
        }
    }
}