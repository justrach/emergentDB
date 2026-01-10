//! QD Engine - Quality-Diversity optimization via MAP-Elites.
//!
//! This crate provides:
//! - MAP-Elites algorithm implementation
//! - Behavior characterization for queries
//! - Archive management with grid-based niching
//! - Parallel evolution with elitism

pub mod archive;
pub mod behavior;
pub mod evolution;

pub use archive::Archive;
pub use behavior::BehaviorDescriptor;
pub use evolution::MapElites;
