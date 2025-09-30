//! Operations module - CPU and GPU implementations

pub mod cpu;

// Re-export commonly used functions
pub use cpu::{
    assign_points,
    compute_robust_stats,
    update_anchors,
    compute_energy,
    count_assignment_changes,
};