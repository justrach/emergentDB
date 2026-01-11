//! Code index for fast file discovery and caching

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::time::SystemTime;
use ignore::WalkBuilder;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub size: u64,
    pub modified: u64,
    pub extension: Option<String>,
    pub line_count: usize,
}

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub root: PathBuf,
    pub respect_gitignore: bool,
    pub max_file_size: u64,
    pub include_hidden: bool,
    pub file_types: Option<Vec<String>>,
    pub exclude_patterns: Vec<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            respect_gitignore: true,
            max_file_size: 10 * 1024 * 1024, // 10MB
            include_hidden: false,
            file_types: None,
            exclude_patterns: vec![
                "*.lock".into(),
                "*.min.js".into(),
                "*.min.css".into(),
                "node_modules".into(),
                "target".into(),
                ".git".into(),
                "dist".into(),
                "build".into(),
            ],
        }
    }
}

pub struct CodeIndex {
    config: IndexConfig,
    files: RwLock<HashMap<PathBuf, FileEntry>>,
    indexed_at: RwLock<Option<SystemTime>>,
}

impl CodeIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            files: RwLock::new(HashMap::new()),
            indexed_at: RwLock::new(None),
        }
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        Self::new(IndexConfig {
            root: path.as_ref().to_path_buf(),
            ..Default::default()
        })
    }

    /// Build or rebuild the file index
    pub fn build(&self) -> Result<usize> {
        let mut files = HashMap::new();
        let walker = WalkBuilder::new(&self.config.root)
            .hidden(!self.config.include_hidden)
            .git_ignore(self.config.respect_gitignore)
            .build();

        for entry in walker.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Skip excluded patterns
            if self.should_exclude(path) {
                continue;
            }

            // Check file type filter
            if let Some(ref types) = self.config.file_types {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                if !types.iter().any(|t| t == ext) {
                    continue;
                }
            }

            // Get metadata
            let metadata = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Skip large files
            if metadata.len() > self.config.max_file_size {
                continue;
            }

            let modified = metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            // Count lines (quick estimate)
            let line_count = std::fs::read_to_string(path)
                .map(|s| s.lines().count())
                .unwrap_or(0);

            let entry = FileEntry {
                path: path.to_path_buf(),
                size: metadata.len(),
                modified,
                extension: path.extension().and_then(|e| e.to_str()).map(String::from),
                line_count,
            };

            files.insert(path.to_path_buf(), entry);
        }

        let count = files.len();
        *self.files.write() = files;
        *self.indexed_at.write() = Some(SystemTime::now());

        Ok(count)
    }

    fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        for pattern in &self.config.exclude_patterns {
            if pattern.contains('*') {
                // Glob pattern
                if let Ok(glob) = globset::Glob::new(pattern) {
                    let matcher = glob.compile_matcher();
                    if matcher.is_match(path) {
                        return true;
                    }
                }
            } else {
                // Simple contains check
                if path_str.contains(pattern) {
                    return true;
                }
            }
        }
        false
    }

    pub fn files(&self) -> Vec<FileEntry> {
        self.files.read().values().cloned().collect()
    }

    pub fn file_count(&self) -> usize {
        self.files.read().len()
    }

    pub fn get_file(&self, path: &Path) -> Option<FileEntry> {
        self.files.read().get(path).cloned()
    }

    /// Get files by extension
    pub fn files_by_ext(&self, ext: &str) -> Vec<FileEntry> {
        self.files.read()
            .values()
            .filter(|f| f.extension.as_deref() == Some(ext))
            .cloned()
            .collect()
    }
}
