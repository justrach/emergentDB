//! Storage traits and common types for persistence backends.

use crate::{Embedding, NodeId};
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("Failed to open database at {path}: {source}")]
    OpenError {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Failed to read key {key}: {source}")]
    ReadError {
        key: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Failed to write key {key}: {source}")]
    WriteError {
        key: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Failed to delete key {key}: {source}")]
    DeleteError {
        key: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Database is closed")]
    DatabaseClosed,

    #[error("Iterator error: {0}")]
    IteratorError(String),
}

pub type StorageResult<T> = std::result::Result<T, StorageError>;

/// Configuration for storage backends.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Path to the data directory
    pub data_dir: PathBuf,

    /// Whether to enable write-ahead logging (WAL) for durability
    /// If false, writes may be lost on crash but performance is better
    pub enable_wal: bool,

    /// Whether to sync writes to disk immediately
    /// If false, OS may buffer writes (faster but less durable)
    pub sync_writes: bool,

    /// Compression type for stored vectors
    pub compression: CompressionType,

    /// Maximum number of open files for RocksDB
    pub max_open_files: i32,

    /// Write buffer size in bytes (default: 64MB)
    pub write_buffer_size: usize,

    /// Whether to create the database if it doesn't exist
    pub create_if_missing: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./emergentdb_data"),
            enable_wal: true,
            sync_writes: false, // Async for performance
            compression: CompressionType::Lz4,
            max_open_files: 256,
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            create_if_missing: true,
        }
    }
}

impl StorageConfig {
    /// Create a config for maximum durability (slower writes)
    pub fn durable(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            enable_wal: true,
            sync_writes: true,
            compression: CompressionType::Lz4,
            max_open_files: 256,
            write_buffer_size: 64 * 1024 * 1024,
            create_if_missing: true,
        }
    }

    /// Create a config for maximum performance (may lose recent writes on crash)
    pub fn fast(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            enable_wal: false,
            sync_writes: false,
            compression: CompressionType::None,
            max_open_files: 512,
            write_buffer_size: 128 * 1024 * 1024, // 128MB
            create_if_missing: true,
        }
    }
}

/// Compression types supported by the storage backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression (fastest, largest)
    None,
    /// LZ4 compression (fast, good ratio)
    Lz4,
    /// Zstd compression (slower, best ratio)
    Zstd,
    /// Snappy compression (very fast, moderate ratio)
    Snappy,
}

/// Trait for persistent vector storage backends.
///
/// Implementations must be thread-safe (Send + Sync).
pub trait Storage: Send + Sync {
    /// Store a vector with its ID.
    fn put(&self, id: NodeId, embedding: &Embedding) -> StorageResult<()>;

    /// Retrieve a vector by ID.
    fn get(&self, id: NodeId) -> StorageResult<Option<Embedding>>;

    /// Delete a vector by ID.
    fn delete(&self, id: NodeId) -> StorageResult<bool>;

    /// Check if a vector exists.
    fn contains(&self, id: NodeId) -> StorageResult<bool>;

    /// Get the number of stored vectors.
    fn len(&self) -> StorageResult<usize>;

    /// Check if the storage is empty.
    fn is_empty(&self) -> StorageResult<bool> {
        Ok(self.len()? == 0)
    }

    /// Iterate over all stored vectors.
    /// Returns an iterator of (NodeId, Embedding) pairs.
    fn iter(&self) -> StorageResult<Box<dyn Iterator<Item = StorageResult<(NodeId, Embedding)>> + '_>>;

    /// Flush any buffered writes to disk.
    fn flush(&self) -> StorageResult<()>;

    /// Close the storage backend gracefully.
    fn close(&self) -> StorageResult<()>;

    /// Get storage statistics.
    fn stats(&self) -> StorageStats;
}

/// Statistics about the storage backend.
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Number of vectors stored
    pub vector_count: usize,
    /// Approximate size on disk in bytes
    pub disk_size_bytes: u64,
    /// Number of read operations
    pub reads: u64,
    /// Number of write operations
    pub writes: u64,
    /// Number of delete operations
    pub deletes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StorageConfig::default();
        assert!(config.enable_wal);
        assert!(!config.sync_writes);
        assert!(config.create_if_missing);
    }

    #[test]
    fn test_durable_config() {
        let config = StorageConfig::durable(PathBuf::from("/tmp/test"));
        assert!(config.enable_wal);
        assert!(config.sync_writes);
    }

    #[test]
    fn test_fast_config() {
        let config = StorageConfig::fast(PathBuf::from("/tmp/test"));
        assert!(!config.enable_wal);
        assert!(!config.sync_writes);
        assert_eq!(config.compression, CompressionType::None);
    }
}
