//! # Storage Module
//!
//! Persistent storage backends for EmergentDB vectors.
//!
//! This module provides durable storage options that allow vectors to survive
//! server restarts while maintaining in-memory search performance.
//!
//! ## Architecture
//!
//! ```text
//! Search: RAM ──SIMD──→ Results  (fast path, unchanged)
//!
//! Insert: RAM + async write → Disk  (durability)
//!
//! Startup: Disk → RAM  (recovery)
//! ```
//!
//! ## Available Backends
//!
//! - `RocksDbStorage`: High-performance LSM-tree storage using RocksDB
//! - `MemoryStorage`: In-memory only (no persistence, for testing)

mod rocksdb_backend;
mod traits;

pub use rocksdb_backend::RocksDbStorage;
pub use traits::{Storage, StorageConfig, StorageError};
