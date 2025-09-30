// Anchor update kernel - moves anchors toward robust means with temperature scaling
// Based on the spec in docs/KERNELS.md

struct Params {
    m: u32,               // number of anchors
    d: u32,               // dimension
    temperature: f32,     // annealing temperature
    learning_rate: f32,   // base learning rate
    momentum: f32,        // momentum coefficient
    enable_jacobians: u32, // 1 if jacobians enabled, 0 otherwise
}

@group(0) @binding(0) var<storage, read_write> anchors: array<f32>;    // [m × d]
@group(0) @binding(1) var<storage, read> stats: array<f32>;           // [m × (d+2)]
@group(0) @binding(2) var<storage, read_write> jacobians: array<f32>;  // [m × d] optional
@group(0) @binding(3) var<storage, read_write> momentum_buffer: array<f32>; // [m × d] for momentum

@group(1) @binding(0) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
const MAX_STEP: f32 = 0.1; // Maximum step size for stability

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let global_idx = global_id.x;

    // Each thread handles one dimension of one anchor
    let anchor_id = global_idx / params.d;
    let dim_id = global_idx % params.d;

    // Bounds check
    if (anchor_id >= params.m) {
        return;
    }

    // Read statistics for this anchor
    let stats_base = anchor_id * (params.d + 2u);
    let mean_value = stats[stats_base + dim_id];
    let point_count = stats[stats_base + params.d];

    // Skip update if no points assigned
    if (point_count < 1.0) {
        return;
    }

    // Current anchor value
    let anchor_idx = anchor_id * params.d + dim_id;
    let old_value = anchors[anchor_idx];

    // Compute gradient step with temperature scaling
    let gradient = mean_value - old_value;
    let temp_scale = params.temperature / (1.0 + params.temperature);
    var step = gradient * params.learning_rate * temp_scale;

    // Apply momentum if enabled
    if (params.momentum > 0.0) {
        let momentum_idx = anchor_idx;
        let old_momentum = momentum_buffer[momentum_idx];
        step = params.momentum * old_momentum + (1.0 - params.momentum) * step;
        momentum_buffer[momentum_idx] = step;
    }

    // Clamp step size for numerical stability
    step = clamp(step, -MAX_STEP, MAX_STEP);

    // Update anchor
    let new_value = old_value + step;
    anchors[anchor_idx] = new_value;

    // Optional: Update diagonal Jacobian using exponential moving average
    if (params.enable_jacobians == 1u) {
        let local_variance = stats[stats_base + params.d + 1u];
        let jacobian_idx = anchor_idx;

        // Exponential moving average of local gradient magnitude
        let current_jacobian = jacobians[jacobian_idx];
        let gradient_magnitude = sqrt(local_variance + 0.001); // small regularization
        let new_jacobian = 0.9 * current_jacobian + 0.1 * gradient_magnitude;

        jacobians[jacobian_idx] = new_jacobian;
    }
}