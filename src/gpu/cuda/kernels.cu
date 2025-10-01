// VLC CUDA Kernels
// Ported from WGSL shaders for NVIDIA GPU acceleration
// Targeting sm_53+ (Jetson Nano through RTX 40xx)

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// ASSIGN KERNEL - Assigns each point to the nearest anchor
// ============================================================================

struct AssignParams {
    uint32_t n;  // number of points
    uint32_t m;  // number of anchors
    uint32_t d;  // dimension
};

extern "C" __global__ void assign_kernel(
    const float* points,     // [n × d]
    const float* anchors,    // [m × d]
    uint32_t* assigns,       // [n]
    AssignParams params
) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (thread_id >= params.n) {
        return;
    }

    uint32_t point_offset = thread_id * params.d;
    float min_dist = 3.4028235e38f; // f32::INFINITY
    uint32_t min_anchor = 0;

    // Find nearest anchor
    for (uint32_t anchor_id = 0; anchor_id < params.m; anchor_id++) {
        uint32_t anchor_offset = anchor_id * params.d;
        float dist = 0.0f;

        // Vectorized distance computation (4 elements at a time)
        uint32_t full_chunks = params.d / 4;
        for (uint32_t chunk = 0; chunk < full_chunks; chunk++) {
            uint32_t base_idx = chunk * 4;
            uint32_t p_idx = point_offset + base_idx;
            uint32_t a_idx = anchor_offset + base_idx;

            float diff0 = points[p_idx + 0] - anchors[a_idx + 0];
            float diff1 = points[p_idx + 1] - anchors[a_idx + 1];
            float diff2 = points[p_idx + 2] - anchors[a_idx + 2];
            float diff3 = points[p_idx + 3] - anchors[a_idx + 3];

            dist += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
        }

        // Handle remaining dimensions
        uint32_t remainder_start = full_chunks * 4;
        for (uint32_t dim = remainder_start; dim < params.d; dim++) {
            float diff = points[point_offset + dim] - anchors[anchor_offset + dim];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_anchor = anchor_id;
        }
    }

    assigns[thread_id] = min_anchor;
}

// ============================================================================
// REDUCE KERNEL - Computes robust statistics for each anchor
// ============================================================================

struct ReduceParams {
    uint32_t n;              // number of points
    uint32_t m;              // number of anchors
    uint32_t d;              // dimension
    float huber_threshold;   // threshold for robust estimation
};

// Shared memory declarations (declared dynamically at kernel launch)
extern __shared__ float shared_mem[];

extern "C" __global__ void reduce_kernel(
    const float* points,     // [n × d]
    const uint32_t* assigns, // [n]
    float* stats,            // [m × (d+2)]
    ReduceParams params
) {
    uint32_t anchor_id = blockIdx.x;
    uint32_t thread_id = threadIdx.x;
    const uint32_t WORKGROUP_SIZE = blockDim.x;

    // Bounds check
    if (anchor_id >= params.m) {
        return;
    }

    // Shared memory layout: [partial_sums[256], total_weight[1]]
    float* partial_sums = shared_mem;
    uint32_t* total_weight = (uint32_t*)&shared_mem[WORKGROUP_SIZE];

    // Initialize total_weight
    if (thread_id == 0) {
        *total_weight = 0;
    }
    __syncthreads();

    // Step 1: Compute robust mean for each dimension
    for (uint32_t dim = 0; dim < params.d; dim++) {
        // Initialize partial sum for this dimension
        partial_sums[thread_id] = 0.0f;
        __syncthreads();

        // Each thread processes multiple points
        for (uint32_t point_id = thread_id; point_id < params.n; point_id += WORKGROUP_SIZE) {
            if (assigns[point_id] == anchor_id) {
                float point_value = points[point_id * params.d + dim];

                // Simplified Huber weighting (unit weight for now)
                float weight = 1.0f;

                partial_sums[thread_id] += point_value * weight;

                // Count points only in first dimension to avoid double counting
                if (dim == 0) {
                    atomicAdd(total_weight, 1);
                }
            }
        }

        __syncthreads();

        // Tree reduction of partial sums
        for (uint32_t active_threads = WORKGROUP_SIZE / 2; active_threads > 0; active_threads /= 2) {
            if (thread_id < active_threads) {
                partial_sums[thread_id] += partial_sums[thread_id + active_threads];
            }
            __syncthreads();
        }

        // Thread 0 writes the mean for this dimension
        if (thread_id == 0) {
            float weight_f32 = (float)(*total_weight);
            if (weight_f32 > 0.0f) {
                stats[anchor_id * (params.d + 2) + dim] = partial_sums[0] / weight_f32;
            } else {
                stats[anchor_id * (params.d + 2) + dim] = 0.0f;
            }
        }

        __syncthreads();
    }

    // Step 2: Write count and compute simple variance
    if (thread_id == 0) {
        float weight_f32 = (float)(*total_weight);
        stats[anchor_id * (params.d + 2) + params.d] = weight_f32;

        // Simple variance computation
        float total_variance = 0.0f;
        for (uint32_t point_id = 0; point_id < params.n; point_id++) {
            if (assigns[point_id] == anchor_id) {
                float point_variance = 0.0f;
                for (uint32_t dim = 0; dim < params.d; dim++) {
                    float point_val = points[point_id * params.d + dim];
                    float mean_val = stats[anchor_id * (params.d + 2) + dim];
                    float diff = point_val - mean_val;
                    point_variance += diff * diff;
                }
                total_variance += point_variance;
            }
        }

        if (weight_f32 > 1.0f) {
            stats[anchor_id * (params.d + 2) + params.d + 1] = total_variance / (weight_f32 - 1.0f);
        } else {
            stats[anchor_id * (params.d + 2) + params.d + 1] = 0.0f;
        }
    }
}

// ============================================================================
// UPDATE KERNEL - Moves anchors toward robust means with temperature scaling
// ============================================================================

struct UpdateParams {
    uint32_t m;              // number of anchors
    uint32_t d;              // dimension
    float temperature;       // annealing temperature
    float learning_rate;     // base learning rate
    float momentum;          // momentum coefficient
    uint32_t enable_jacobians; // 1 if jacobians enabled, 0 otherwise
};

const float MAX_STEP = 0.1f; // Maximum step size for stability

extern "C" __global__ void update_kernel(
    float* anchors,              // [m × d]
    const float* stats,          // [m × (d+2)]
    float* jacobians,            // [m × d] optional
    float* momentum_buffer,      // [m × d] for momentum
    UpdateParams params
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one dimension of one anchor
    uint32_t anchor_id = global_idx / params.d;
    uint32_t dim_id = global_idx % params.d;

    // Bounds check
    if (anchor_id >= params.m) {
        return;
    }

    // Read statistics for this anchor
    uint32_t stats_base = anchor_id * (params.d + 2);
    float mean_value = stats[stats_base + dim_id];
    float point_count = stats[stats_base + params.d];

    // Skip update if no points assigned
    if (point_count < 1.0f) {
        return;
    }

    // Current anchor value
    uint32_t anchor_idx = anchor_id * params.d + dim_id;
    float old_value = anchors[anchor_idx];

    // Compute gradient step with temperature scaling
    float gradient = mean_value - old_value;
    float temp_scale = params.temperature / (1.0f + params.temperature);
    float step = gradient * params.learning_rate * temp_scale;

    // Apply momentum if enabled
    if (params.momentum > 0.0f) {
        uint32_t momentum_idx = anchor_idx;
        float old_momentum = momentum_buffer[momentum_idx];
        step = params.momentum * old_momentum + (1.0f - params.momentum) * step;
        momentum_buffer[momentum_idx] = step;
    }

    // Clamp step size for numerical stability
    step = fminf(fmaxf(step, -MAX_STEP), MAX_STEP);

    // Update anchor
    float new_value = old_value + step;
    anchors[anchor_idx] = new_value;

    // Optional: Update diagonal Jacobian using exponential moving average
    if (params.enable_jacobians == 1) {
        float local_variance = stats[stats_base + params.d + 1];
        uint32_t jacobian_idx = anchor_idx;

        // Exponential moving average of local gradient magnitude
        float current_jacobian = jacobians[jacobian_idx];
        float gradient_magnitude = sqrtf(local_variance + 0.001f); // small regularization
        float new_jacobian = 0.9f * current_jacobian + 0.1f * gradient_magnitude;

        jacobians[jacobian_idx] = new_jacobian;
    }
}
