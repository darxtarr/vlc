// Point assignment kernel - assigns each point to nearest anchor
// Based on the spec in docs/KERNELS.md

struct Params {
    n: u32,  // number of points
    m: u32,  // number of anchors
    d: u32,  // dimension
}

@group(0) @binding(0) var<storage, read> points: array<f32>;     // [n × d]
@group(0) @binding(1) var<storage, read> anchors: array<f32>;    // [m × d]
@group(0) @binding(2) var<storage, read_write> assigns: array<u32>; // [n]

@group(1) @binding(0) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    // Bounds check
    if (thread_id >= params.n) {
        return;
    }

    let point_offset = thread_id * params.d;
    var min_dist = 3.4028235e38; // f32::INFINITY
    var min_anchor = 0u;

    // Find nearest anchor
    for (var anchor_id = 0u; anchor_id < params.m; anchor_id++) {
        let anchor_offset = anchor_id * params.d;
        var dist = 0.0;

        // Vectorized distance computation (4 elements at a time)
        let full_chunks = params.d / 4u;
        for (var chunk = 0u; chunk < full_chunks; chunk++) {
            let base_idx = chunk * 4u;
            let p_idx = point_offset + base_idx;
            let a_idx = anchor_offset + base_idx;

            let diff0 = points[p_idx + 0u] - anchors[a_idx + 0u];
            let diff1 = points[p_idx + 1u] - anchors[a_idx + 1u];
            let diff2 = points[p_idx + 2u] - anchors[a_idx + 2u];
            let diff3 = points[p_idx + 3u] - anchors[a_idx + 3u];

            dist += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
        }

        // Handle remaining dimensions
        let remainder_start = full_chunks * 4u;
        for (var dim = remainder_start; dim < params.d; dim++) {
            let diff = points[point_offset + dim] - anchors[anchor_offset + dim];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_anchor = anchor_id;
        }
    }

    assigns[thread_id] = min_anchor;
}