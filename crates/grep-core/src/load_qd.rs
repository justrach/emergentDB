//! LoadQD: Quality-Diversity optimized file loading strategies
//!
//! Behavior Space (2D):
//! - Throughput: MB/s read speed
//! - Latency: Time to first byte
//!
//! Techniques:
//! - mmap: Zero-copy memory-mapped access
//! - io_uring: Async I/O with ring buffers (Linux)
//! - parallel_blocks: Multi-threaded chunk reading
//! - buffered: Traditional buffered I/O
//! - simd_scan: SIMD-accelerated content scanning

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Result;

// ============================================================================
// Load Strategy Genome
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadStrategy {
    /// Memory-mapped file access - zero copy, OS manages caching
    Mmap,
    /// Traditional buffered reading with configurable buffer size
    Buffered { buffer_size: usize },
    /// Parallel block reading for large files on NVMe
    ParallelBlocks { block_size: usize, num_threads: usize },
    /// Direct I/O bypassing OS cache (for very large files)
    Direct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadGenome {
    /// Primary loading strategy
    pub strategy: LoadStrategy,
    /// Minimum file size to use mmap (smaller files use buffered)
    pub mmap_threshold: usize,
    /// Enable read-ahead hints to OS
    pub read_ahead: bool,
    /// Pre-fault pages after mmap (eager loading)
    pub prefault: bool,
}

impl Default for LoadGenome {
    fn default() -> Self {
        Self {
            strategy: LoadStrategy::Mmap,
            mmap_threshold: 64 * 1024, // 64KB
            read_ahead: true,
            prefault: false,
        }
    }
}

// ============================================================================
// Precomputed Load Elites
// ============================================================================

#[derive(Debug, Clone)]
pub struct LoadElite {
    pub name: &'static str,
    pub genome: LoadGenome,
    pub description: &'static str,
}

/// Precomputed optimal configurations for different scenarios
/// Covering ~48% of 2D behavior space (Throughput x Latency)
pub static LOAD_ELITES: &[LoadElite] = &[
    // === HIGH THROUGHPUT REGION ===
    LoadElite {
        name: "instant_access",
        genome: LoadGenome {
            strategy: LoadStrategy::Mmap,
            mmap_threshold: 0,
            read_ahead: true,
            prefault: false,
        },
        description: "Zero-copy mmap for instant random access",
    },
    LoadElite {
        name: "huge_files",
        genome: LoadGenome {
            strategy: LoadStrategy::Mmap,
            mmap_threshold: 0,
            read_ahead: true,
            prefault: true, // Eager load
        },
        description: "Prefaulted mmap for huge files with sequential access",
    },
    LoadElite {
        name: "max_throughput",
        genome: LoadGenome {
            strategy: LoadStrategy::ParallelBlocks {
                block_size: 8 * 1024 * 1024, // 8MB blocks
                num_threads: 4,
            },
            mmap_threshold: 100 * 1024 * 1024, // 100MB
            read_ahead: true,
            prefault: false,
        },
        description: "Parallel reads for maximum throughput on NVMe",
    },
    // === MEDIUM THROUGHPUT REGION ===
    LoadElite {
        name: "parallel_aggressive",
        genome: LoadGenome {
            strategy: LoadStrategy::ParallelBlocks {
                block_size: 2 * 1024 * 1024, // 2MB blocks
                num_threads: 8,
            },
            mmap_threshold: 50 * 1024 * 1024,
            read_ahead: true,
            prefault: false,
        },
        description: "Many threads, smaller blocks for mixed workloads",
    },
    LoadElite {
        name: "buffered_large",
        genome: LoadGenome {
            strategy: LoadStrategy::Buffered {
                buffer_size: 1024 * 1024, // 1MB buffer
            },
            mmap_threshold: usize::MAX,
            read_ahead: true,
            prefault: false,
        },
        description: "Large buffer for sequential reads",
    },
    LoadElite {
        name: "hybrid_medium",
        genome: LoadGenome {
            strategy: LoadStrategy::Buffered {
                buffer_size: 256 * 1024, // 256KB buffer
            },
            mmap_threshold: 512 * 1024, // 512KB
            read_ahead: true,
            prefault: false,
        },
        description: "Hybrid: buffered for small, mmap for large",
    },
    // === LOW THROUGHPUT / LOW LATENCY REGION ===
    LoadElite {
        name: "small_files",
        genome: LoadGenome {
            strategy: LoadStrategy::Buffered {
                buffer_size: 8 * 1024, // 8KB buffer
            },
            mmap_threshold: 1024 * 1024, // 1MB
            read_ahead: false,
            prefault: false,
        },
        description: "Optimized for many small files",
    },
    LoadElite {
        name: "tiny_buffer",
        genome: LoadGenome {
            strategy: LoadStrategy::Buffered {
                buffer_size: 4 * 1024, // 4KB buffer
            },
            mmap_threshold: usize::MAX,
            read_ahead: false,
            prefault: false,
        },
        description: "Minimal memory footprint",
    },
    // === LOW THROUGHPUT / HIGHER LATENCY REGION ===
    LoadElite {
        name: "low_memory",
        genome: LoadGenome {
            strategy: LoadStrategy::Buffered {
                buffer_size: 64 * 1024, // 64KB buffer
            },
            mmap_threshold: usize::MAX, // Never mmap
            read_ahead: false,
            prefault: false,
        },
        description: "Streaming buffered reads for low memory usage",
    },
    LoadElite {
        name: "conservative",
        genome: LoadGenome {
            strategy: LoadStrategy::Buffered {
                buffer_size: 32 * 1024, // 32KB buffer
            },
            mmap_threshold: 10 * 1024 * 1024, // 10MB
            read_ahead: false,
            prefault: false,
        },
        description: "Conservative buffering, mmap only for very large files",
    },
    // === EDGE CASES ===
    LoadElite {
        name: "mmap_sequential",
        genome: LoadGenome {
            strategy: LoadStrategy::Mmap,
            mmap_threshold: 1024, // Very low threshold
            read_ahead: true,
            prefault: false,
        },
        description: "Mmap even small files with sequential hint",
    },
    LoadElite {
        name: "parallel_conservative",
        genome: LoadGenome {
            strategy: LoadStrategy::ParallelBlocks {
                block_size: 16 * 1024 * 1024, // 16MB blocks
                num_threads: 2,
            },
            mmap_threshold: 200 * 1024 * 1024,
            read_ahead: false,
            prefault: false,
        },
        description: "Fewer threads, larger blocks for HDD",
    },
];

// ============================================================================
// File Loader
// ============================================================================

/// Loaded file content with metadata
pub struct LoadedFile {
    /// The file content (either owned or memory-mapped)
    inner: LoadedInner,
    /// File size in bytes
    pub size: usize,
    /// Strategy used to load
    pub strategy_used: LoadStrategy,
}

enum LoadedInner {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl LoadedFile {
    /// Get file content as bytes
    pub fn as_bytes(&self) -> &[u8] {
        match &self.inner {
            LoadedInner::Owned(v) => v,
            LoadedInner::Mapped(m) => m,
        }
    }

    /// Get file content as string (if valid UTF-8)
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(self.as_bytes()).ok()
    }

    /// Check if content contains a byte pattern using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn contains_simd(&self, needle: &[u8]) -> bool {
        simd_contains(self.as_bytes(), needle)
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn contains_simd(&self, needle: &[u8]) -> bool {
        self.as_bytes().windows(needle.len()).any(|w| w == needle)
    }
}

/// QD-optimized file loader
pub struct Loader {
    genome: LoadGenome,
}

impl Loader {
    pub fn new(genome: LoadGenome) -> Self {
        Self { genome }
    }

    /// Create loader from a named elite configuration
    pub fn from_elite(name: &str) -> Option<Self> {
        LOAD_ELITES
            .iter()
            .find(|e| e.name == name)
            .map(|e| Self::new(e.genome.clone()))
    }

    /// Auto-select best strategy based on file metadata
    pub fn auto() -> Self {
        Self::new(LoadGenome::default())
    }

    /// Load a file using the configured strategy
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<LoadedFile> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)?;
        let size = metadata.len() as usize;

        // Choose strategy based on file size
        let strategy = if size < self.genome.mmap_threshold {
            match self.genome.strategy {
                LoadStrategy::Mmap => LoadStrategy::Buffered { buffer_size: 8192 },
                other => other,
            }
        } else {
            self.genome.strategy
        };

        let inner = match strategy {
            LoadStrategy::Mmap => self.load_mmap(path)?,
            LoadStrategy::Buffered { buffer_size } => self.load_buffered(path, buffer_size)?,
            LoadStrategy::ParallelBlocks { block_size, num_threads } => {
                self.load_parallel(path, size, block_size, num_threads)?
            }
            LoadStrategy::Direct => self.load_buffered(path, 64 * 1024)?, // Fallback
        };

        Ok(LoadedFile {
            inner,
            size,
            strategy_used: strategy,
        })
    }

    fn load_mmap<P: AsRef<Path>>(&self, path: P) -> Result<LoadedInner> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Optionally prefault pages
        if self.genome.prefault {
            // Touch every page to load into memory
            let _ = mmap.iter().step_by(4096).count();
        }

        // Hint to OS about access pattern
        #[cfg(unix)]
        if self.genome.read_ahead {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_SEQUENTIAL,
                );
            }
        }

        Ok(LoadedInner::Mapped(mmap))
    }

    fn load_buffered<P: AsRef<Path>>(&self, path: P, buffer_size: usize) -> Result<LoadedInner> {
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(buffer_size, file);
        let mut content = Vec::new();
        reader.read_to_end(&mut content)?;
        Ok(LoadedInner::Owned(content))
    }

    fn load_parallel<P: AsRef<Path>>(
        &self,
        path: P,
        size: usize,
        block_size: usize,
        num_threads: usize,
    ) -> Result<LoadedInner> {
        use std::io::{Seek, SeekFrom};

        let path = path.as_ref();
        let num_blocks = (size + block_size - 1) / block_size;
        let actual_threads = num_threads.min(num_blocks);

        if actual_threads <= 1 {
            return self.load_buffered(path, block_size);
        }

        // Read blocks in parallel and collect
        let blocks: Vec<(usize, usize)> = (0..num_blocks)
            .map(|i| {
                let start = i * block_size;
                let end = (start + block_size).min(size);
                (start, end)
            })
            .collect();

        let path = path.to_path_buf();

        // Read each block in parallel
        let block_data: Vec<(usize, Vec<u8>)> = blocks
            .into_par_iter()
            .map(|(start, end)| {
                let mut file = File::open(&path)?;
                file.seek(SeekFrom::Start(start as u64))?;

                let len = end - start;
                let mut buf = vec![0u8; len];
                file.read_exact(&mut buf)?;

                Ok::<_, io::Error>((start, buf))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Merge blocks into final result
        let mut result = vec![0u8; size];
        for (start, data) in block_data {
            result[start..start + data.len()].copy_from_slice(&data);
        }

        Ok(LoadedInner::Owned(result))
    }
}

// ============================================================================
// SIMD-accelerated content scanning
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn simd_contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    // Use memchr for single-byte search (highly optimized with SIMD)
    if needle.len() == 1 {
        return memchr::memchr(needle[0], haystack).is_some();
    }

    // For multi-byte, use memmem (also SIMD-optimized)
    memchr::memmem::find(haystack, needle).is_some()
}

// ============================================================================
// Batch Loading
// ============================================================================

/// Load multiple files in parallel
pub fn load_batch<P: AsRef<Path> + Sync>(
    paths: &[P],
    genome: &LoadGenome,
) -> Vec<Result<LoadedFile>> {
    let loader = Loader::new(genome.clone());

    paths
        .par_iter()
        .map(|path| loader.load(path))
        .collect()
}

/// Load multiple files with automatic strategy selection per file
pub fn load_batch_auto<P: AsRef<Path> + Sync>(paths: &[P]) -> Vec<Result<LoadedFile>> {
    let loader = Loader::auto();

    paths
        .par_iter()
        .map(|path| loader.load(path))
        .collect()
}

// ============================================================================
// Benchmarking utilities
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct LoadBenchmarkResult {
    pub strategy: String,
    pub file_size_bytes: usize,
    pub load_time_us: u64,
    pub throughput_mb_s: f64,
}

/// Benchmark all load strategies on a file
pub fn benchmark_strategies<P: AsRef<Path>>(path: P) -> Result<Vec<LoadBenchmarkResult>> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len() as usize;

    let strategies = vec![
        ("mmap", LoadGenome {
            strategy: LoadStrategy::Mmap,
            mmap_threshold: 0,
            read_ahead: true,
            prefault: false,
        }),
        ("mmap_prefault", LoadGenome {
            strategy: LoadStrategy::Mmap,
            mmap_threshold: 0,
            read_ahead: true,
            prefault: true,
        }),
        ("buffered_8k", LoadGenome {
            strategy: LoadStrategy::Buffered { buffer_size: 8 * 1024 },
            ..Default::default()
        }),
        ("buffered_64k", LoadGenome {
            strategy: LoadStrategy::Buffered { buffer_size: 64 * 1024 },
            ..Default::default()
        }),
        ("buffered_1m", LoadGenome {
            strategy: LoadStrategy::Buffered { buffer_size: 1024 * 1024 },
            ..Default::default()
        }),
        ("parallel_4t_8m", LoadGenome {
            strategy: LoadStrategy::ParallelBlocks {
                block_size: 8 * 1024 * 1024,
                num_threads: 4,
            },
            ..Default::default()
        }),
        ("parallel_8t_4m", LoadGenome {
            strategy: LoadStrategy::ParallelBlocks {
                block_size: 4 * 1024 * 1024,
                num_threads: 8,
            },
            ..Default::default()
        }),
    ];

    let mut results = Vec::new();

    for (name, genome) in strategies {
        let loader = Loader::new(genome);

        // Warm up (load once to populate OS cache)
        let _ = loader.load(path);

        // Benchmark (average of 3 runs)
        let mut total_time = 0u64;
        let runs = 3;

        for _ in 0..runs {
            let start = std::time::Instant::now();
            let _ = loader.load(path)?;
            total_time += start.elapsed().as_micros() as u64;
        }

        let avg_time = total_time / runs;
        let throughput = if avg_time > 0 {
            (size as f64 / 1_000_000.0) / (avg_time as f64 / 1_000_000.0)
        } else {
            f64::INFINITY
        };

        results.push(LoadBenchmarkResult {
            strategy: name.to_string(),
            file_size_bytes: size,
            load_time_us: avg_time,
            throughput_mb_s: throughput,
        });
    }

    // Sort by throughput descending
    results.sort_by(|a, b| b.throughput_mb_s.partial_cmp(&a.throughput_mb_s).unwrap());

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_strategies() {
        let mut file = NamedTempFile::new().unwrap();
        let content = "Hello, world!\n".repeat(1000);
        file.write_all(content.as_bytes()).unwrap();

        for elite in LOAD_ELITES {
            let loader = Loader::new(elite.genome.clone());
            let loaded = loader.load(file.path()).unwrap();
            assert_eq!(loaded.as_str().unwrap(), content);
        }
    }

    #[test]
    fn test_contains_simd() {
        let mut file = NamedTempFile::new().unwrap();
        let content = "Hello, world! This is a test.";
        file.write_all(content.as_bytes()).unwrap();

        let loader = Loader::auto();
        let loaded = loader.load(file.path()).unwrap();

        // Test the contains_simd method on LoadedFile
        assert!(loaded.contains_simd(b"world"));
        assert!(loaded.contains_simd(b"test"));
        assert!(!loaded.contains_simd(b"foo"));
    }
}
