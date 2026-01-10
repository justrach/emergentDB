//! RocksDB storage backend for persistent vector storage.
//!
//! This backend provides high-performance persistent storage using RocksDB,
//! an LSM-tree based key-value store optimized for fast writes and reads.
//!
//! ## Key Format
//! - Keys are 8-byte big-endian encoded NodeId (u64)
//! - This ensures lexicographic ordering matches numeric ordering
//!
//! ## Value Format
//! - Values are raw f32 bytes (dimension * 4 bytes)
//! - No additional encoding for maximum performance

use crate::{Embedding, NodeId};
use rocksdb::{IteratorMode, Options, WriteBatch, DB};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;

use super::traits::{CompressionType, Storage, StorageConfig, StorageError, StorageResult, StorageStats};

/// RocksDB-backed persistent storage for vectors.
///
/// Thread-safe and optimized for high-throughput vector storage.
/// Uses RwLock for thread-safety since RocksDB's Rust bindings default to single-threaded.
pub struct RocksDbStorage {
    db: Arc<RwLock<DB>>,
    closed: AtomicBool,
    stats: StorageStatsInner,
}

struct StorageStatsInner {
    reads: AtomicU64,
    writes: AtomicU64,
    deletes: AtomicU64,
}

impl RocksDbStorage {
    /// Open or create a RocksDB storage at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> StorageResult<Self> {
        Self::open_with_config(path, StorageConfig::default())
    }

    /// Open or create a RocksDB storage with custom configuration.
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: StorageConfig) -> StorageResult<Self> {
        let path = path.as_ref();

        let mut opts = Options::default();
        opts.create_if_missing(config.create_if_missing);
        opts.set_max_open_files(config.max_open_files);
        opts.set_write_buffer_size(config.write_buffer_size);

        // Set compression
        match config.compression {
            CompressionType::None => {
                opts.set_compression_type(rocksdb::DBCompressionType::None);
            }
            CompressionType::Lz4 => {
                opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            }
            CompressionType::Zstd => {
                opts.set_compression_type(rocksdb::DBCompressionType::Zstd);
            }
            CompressionType::Snappy => {
                opts.set_compression_type(rocksdb::DBCompressionType::Snappy);
            }
        }

        // Optimize for vector workloads
        opts.set_level_compaction_dynamic_level_bytes(true);
        opts.set_bytes_per_sync(1024 * 1024); // 1MB sync intervals

        // Increase parallelism
        opts.increase_parallelism(num_cpus::get() as i32);

        // Optimize for point lookups (common for vector retrieval)
        opts.optimize_for_point_lookup(64); // 64MB block cache

        let db = DB::open(&opts, path).map_err(|e| StorageError::OpenError {
            path: path.to_path_buf(),
            source: Box::new(e),
        })?;

        Ok(Self {
            db: Arc::new(RwLock::new(db)),
            closed: AtomicBool::new(false),
            stats: StorageStatsInner {
                reads: AtomicU64::new(0),
                writes: AtomicU64::new(0),
                deletes: AtomicU64::new(0),
            },
        })
    }

    /// Encode a NodeId as bytes for use as a key.
    #[inline]
    fn encode_key(id: NodeId) -> [u8; 8] {
        id.0.to_be_bytes()
    }

    /// Decode a key back to a NodeId.
    #[inline]
    fn decode_key(bytes: &[u8]) -> Option<NodeId> {
        if bytes.len() != 8 {
            return None;
        }
        let arr: [u8; 8] = bytes.try_into().ok()?;
        Some(NodeId(u64::from_be_bytes(arr)))
    }

    /// Encode an embedding as bytes.
    #[inline]
    fn encode_value(embedding: &Embedding) -> Vec<u8> {
        // Convert f32 slice to bytes
        let slice = embedding.as_slice();
        let byte_len = slice.len() * std::mem::size_of::<f32>();
        let mut bytes = Vec::with_capacity(byte_len);

        for &val in slice {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        bytes
    }

    /// Decode bytes back to an embedding.
    #[inline]
    fn decode_value(bytes: &[u8]) -> Option<Embedding> {
        if bytes.len() % 4 != 0 {
            return None;
        }

        let dim = bytes.len() / 4;
        let mut data = Vec::with_capacity(dim);

        for chunk in bytes.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().ok()?;
            data.push(f32::from_le_bytes(arr));
        }

        Some(Embedding::new(data))
    }

    /// Batch insert multiple vectors efficiently.
    pub fn put_batch(&self, items: &[(NodeId, Embedding)]) -> StorageResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let mut batch = WriteBatch::default();

        for (id, embedding) in items {
            let key = Self::encode_key(*id);
            let value = Self::encode_value(embedding);
            batch.put(key, value);
        }

        let db = self.db.write();
        db.write(batch).map_err(|e| StorageError::WriteError {
            key: "batch".to_string(),
            source: Box::new(e),
        })?;

        self.stats.writes.fetch_add(items.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Get the approximate size of the database on disk.
    pub fn disk_size(&self) -> u64 {
        let db = self.db.read();
        db.property_int_value("rocksdb.total-sst-files-size")
            .ok()
            .flatten()
            .unwrap_or(0)
    }

    /// Compact the database to reclaim space and improve read performance.
    pub fn compact(&self) -> StorageResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let db = self.db.write();
        db.compact_range::<&[u8], &[u8]>(None, None);
        Ok(())
    }
}

impl Storage for RocksDbStorage {
    fn put(&self, id: NodeId, embedding: &Embedding) -> StorageResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let key = Self::encode_key(id);
        let value = Self::encode_value(embedding);

        let db = self.db.write();
        db.put(key, value).map_err(|e| StorageError::WriteError {
            key: format!("{}", id.0),
            source: Box::new(e),
        })?;

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn get(&self, id: NodeId) -> StorageResult<Option<Embedding>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let key = Self::encode_key(id);

        self.stats.reads.fetch_add(1, Ordering::Relaxed);

        let db = self.db.read();
        match db.get(key) {
            Ok(Some(bytes)) => Self::decode_value(&bytes)
                .ok_or_else(|| StorageError::SerializationError("Invalid embedding data".to_string()))
                .map(Some),
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::ReadError {
                key: format!("{}", id.0),
                source: Box::new(e),
            }),
        }
    }

    fn delete(&self, id: NodeId) -> StorageResult<bool> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let key = Self::encode_key(id);
        let db = self.db.write();

        // Check if exists first
        let exists = db.get(&key).map_err(|e| StorageError::ReadError {
            key: format!("{}", id.0),
            source: Box::new(e),
        })?.is_some();

        if exists {
            db.delete(key).map_err(|e| StorageError::DeleteError {
                key: format!("{}", id.0),
                source: Box::new(e),
            })?;
            self.stats.deletes.fetch_add(1, Ordering::Relaxed);
        }

        Ok(exists)
    }

    fn contains(&self, id: NodeId) -> StorageResult<bool> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let key = Self::encode_key(id);

        let db = self.db.read();
        match db.get(key) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => Err(StorageError::ReadError {
                key: format!("{}", id.0),
                source: Box::new(e),
            }),
        }
    }

    fn len(&self) -> StorageResult<usize> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        // Count by iterating (RocksDB doesn't have a fast count)
        // For large databases, consider caching this
        let db = self.db.read();
        let mut count = 0;
        let iter = db.iterator(IteratorMode::Start);
        for _ in iter {
            count += 1;
        }
        Ok(count)
    }

    fn iter(&self) -> StorageResult<Box<dyn Iterator<Item = StorageResult<(NodeId, Embedding)>> + '_>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        // We need to collect results since we can't hold the lock across iteration
        let db = self.db.read();
        let results: Vec<_> = db.iterator(IteratorMode::Start)
            .map(|result| {
                match result {
                    Ok((key, value)) => {
                        let id = Self::decode_key(&key).ok_or_else(|| {
                            StorageError::SerializationError("Invalid key format".to_string())
                        })?;
                        let embedding = Self::decode_value(&value).ok_or_else(|| {
                            StorageError::SerializationError("Invalid embedding data".to_string())
                        })?;
                        Ok((id, embedding))
                    }
                    Err(e) => Err(StorageError::IteratorError(e.to_string())),
                }
            })
            .collect();

        Ok(Box::new(results.into_iter()))
    }

    fn flush(&self) -> StorageResult<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::DatabaseClosed);
        }

        let db = self.db.write();
        db.flush().map_err(|e| StorageError::WriteError {
            key: "flush".to_string(),
            source: Box::new(e),
        })
    }

    fn close(&self) -> StorageResult<()> {
        self.closed.store(true, Ordering::Release);
        // RocksDB handles cleanup on drop
        Ok(())
    }

    fn stats(&self) -> StorageStats {
        StorageStats {
            vector_count: self.len().unwrap_or(0),
            disk_size_bytes: self.disk_size(),
            reads: self.stats.reads.load(Ordering::Relaxed),
            writes: self.stats.writes.load(Ordering::Relaxed),
            deletes: self.stats.deletes.load(Ordering::Relaxed),
        }
    }
}

impl Clone for RocksDbStorage {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
            closed: AtomicBool::new(self.closed.load(Ordering::Acquire)),
            stats: StorageStatsInner {
                reads: AtomicU64::new(self.stats.reads.load(Ordering::Relaxed)),
                writes: AtomicU64::new(self.stats.writes.load(Ordering::Relaxed)),
                deletes: AtomicU64::new(self.stats.deletes.load(Ordering::Relaxed)),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_encode_decode_key() {
        let id = NodeId(12345);
        let encoded = RocksDbStorage::encode_key(id);
        let decoded = RocksDbStorage::decode_key(&encoded).unwrap();
        assert_eq!(id, decoded);
    }

    #[test]
    fn test_encode_decode_value() {
        let embedding = Embedding::new(vec![1.0, 2.0, 3.0, 4.0]);
        let encoded = RocksDbStorage::encode_value(&embedding);
        let decoded = RocksDbStorage::decode_value(&encoded).unwrap();
        assert_eq!(embedding.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_put_get() {
        let dir = tempdir().unwrap();
        let storage = RocksDbStorage::open(dir.path()).unwrap();

        let id = NodeId(1);
        let embedding = Embedding::new(vec![0.1, 0.2, 0.3]);

        storage.put(id, &embedding).unwrap();
        let retrieved = storage.get(id).unwrap().unwrap();

        assert_eq!(embedding.as_slice(), retrieved.as_slice());
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let storage = RocksDbStorage::open(dir.path()).unwrap();

        let id = NodeId(1);
        let embedding = Embedding::new(vec![0.1, 0.2, 0.3]);

        storage.put(id, &embedding).unwrap();
        assert!(storage.contains(id).unwrap());

        let deleted = storage.delete(id).unwrap();
        assert!(deleted);
        assert!(!storage.contains(id).unwrap());

        // Delete non-existent should return false
        let deleted_again = storage.delete(id).unwrap();
        assert!(!deleted_again);
    }

    #[test]
    fn test_iter() {
        let dir = tempdir().unwrap();
        let storage = RocksDbStorage::open(dir.path()).unwrap();

        // Insert multiple vectors
        for i in 0..10 {
            let id = NodeId(i);
            let embedding = Embedding::new(vec![i as f32; 4]);
            storage.put(id, &embedding).unwrap();
        }

        // Iterate and collect
        let items: Vec<_> = storage.iter().unwrap().collect();
        assert_eq!(items.len(), 10);

        // Verify all succeeded
        for item in items {
            assert!(item.is_ok());
        }
    }

    #[test]
    fn test_batch_insert() {
        let dir = tempdir().unwrap();
        let storage = RocksDbStorage::open(dir.path()).unwrap();

        let items: Vec<_> = (0..100)
            .map(|i| (NodeId(i), Embedding::new(vec![i as f32; 8])))
            .collect();

        storage.put_batch(&items).unwrap();

        assert_eq!(storage.len().unwrap(), 100);
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write data
        {
            let storage = RocksDbStorage::open(&path).unwrap();
            for i in 0..5 {
                storage.put(NodeId(i), &Embedding::new(vec![i as f32; 4])).unwrap();
            }
            storage.flush().unwrap();
        }

        // Reopen and verify
        {
            let storage = RocksDbStorage::open(&path).unwrap();
            assert_eq!(storage.len().unwrap(), 5);

            for i in 0..5 {
                let emb = storage.get(NodeId(i)).unwrap().unwrap();
                assert_eq!(emb.as_slice(), &[i as f32; 4]);
            }
        }
    }
}
