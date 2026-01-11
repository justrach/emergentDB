//! GitHub repository cloning and exploration for LLM agents

use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use serde::{Serialize, Deserialize};

use crate::{Result, GrepError, CodeIndex, IndexConfig, Searcher, SearchQuery, SearchResult};

/// A cloned GitHub repository ready for exploration
pub struct RepoExplorer {
    pub repo_url: String,
    pub local_path: PathBuf,
    pub index: CodeIndex,
    searcher: Option<Searcher>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoInfo {
    pub url: String,
    pub name: String,
    pub local_path: String,
    pub file_count: usize,
    pub indexed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileTree {
    pub path: String,
    pub is_dir: bool,
    pub children: Vec<FileTree>,
    pub size: Option<u64>,
    pub extension: Option<String>,
}

impl RepoExplorer {
    /// Clone a GitHub repo and prepare it for exploration
    ///
    /// # Example
    /// ```ignore
    /// let explorer = RepoExplorer::from_github("https://github.com/user/repo")?;
    /// ```
    pub fn from_github(repo_url: &str) -> Result<Self> {
        let repo_name = Self::extract_repo_name(repo_url)?;
        let cache_dir = Self::cache_dir();
        let local_path = cache_dir.join(&repo_name);

        // Create cache directory
        fs::create_dir_all(&cache_dir).map_err(|e| GrepError::Io(e))?;

        // Clone or pull
        if local_path.exists() {
            // Pull latest
            Self::git_pull(&local_path)?;
        } else {
            // Fresh clone
            Self::git_clone(repo_url, &local_path)?;
        }

        let index = CodeIndex::new(IndexConfig {
            root: local_path.clone(),
            ..Default::default()
        });

        Ok(Self {
            repo_url: repo_url.to_string(),
            local_path,
            index,
            searcher: None,
        })
    }

    /// Load an already cloned repo from local path
    pub fn from_local<P: AsRef<Path>>(path: P) -> Result<Self> {
        let local_path = path.as_ref().to_path_buf();
        if !local_path.exists() {
            return Err(GrepError::PathNotFound(local_path.to_string_lossy().to_string()));
        }

        let index = CodeIndex::new(IndexConfig {
            root: local_path.clone(),
            ..Default::default()
        });

        Ok(Self {
            repo_url: String::new(),
            local_path,
            index,
            searcher: None,
        })
    }

    /// Build the search index
    pub fn build_index(&mut self) -> Result<usize> {
        let count = self.index.build()?;
        self.searcher = Some(Searcher::new(CodeIndex::new(IndexConfig {
            root: self.local_path.clone(),
            ..Default::default()
        })));
        self.searcher.as_mut().unwrap().index.build()?;
        Ok(count)
    }

    /// Get repository info
    pub fn info(&self) -> RepoInfo {
        RepoInfo {
            url: self.repo_url.clone(),
            name: self.local_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            local_path: self.local_path.to_string_lossy().to_string(),
            file_count: self.index.file_count(),
            indexed: self.searcher.is_some(),
        }
    }

    /// Search the repository
    pub fn search(&self, pattern: &str) -> Result<Vec<SearchResult>> {
        let searcher = self.searcher.as_ref()
            .ok_or_else(|| GrepError::IndexError("Index not built. Call build_index() first".into()))?;
        searcher.search(&SearchQuery::literal(pattern))
    }

    /// Search with regex
    pub fn search_regex(&self, pattern: &str) -> Result<Vec<SearchResult>> {
        let searcher = self.searcher.as_ref()
            .ok_or_else(|| GrepError::IndexError("Index not built. Call build_index() first".into()))?;
        searcher.search(&SearchQuery::regex(pattern))
    }

    /// Search with custom query
    pub fn search_advanced(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let searcher = self.searcher.as_ref()
            .ok_or_else(|| GrepError::IndexError("Index not built. Call build_index() first".into()))?;
        searcher.search(query)
    }

    /// Read a specific file
    pub fn read_file(&self, relative_path: &str) -> Result<String> {
        let full_path = self.local_path.join(relative_path);
        fs::read_to_string(&full_path).map_err(|e| GrepError::Io(e))
    }

    /// Read file with line range
    pub fn read_file_lines(&self, relative_path: &str, start: usize, end: usize) -> Result<Vec<String>> {
        let content = self.read_file(relative_path)?;
        let lines: Vec<String> = content.lines()
            .skip(start.saturating_sub(1))
            .take(end - start.saturating_sub(1) + 1)
            .map(String::from)
            .collect();
        Ok(lines)
    }

    /// List files in a directory
    pub fn list_dir(&self, relative_path: &str) -> Result<Vec<String>> {
        let full_path = if relative_path.is_empty() || relative_path == "." {
            self.local_path.clone()
        } else {
            self.local_path.join(relative_path)
        };

        let mut entries = Vec::new();
        for entry in fs::read_dir(&full_path).map_err(|e| GrepError::Io(e))? {
            let entry = entry.map_err(|e| GrepError::Io(e))?;
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.path().is_dir() {
                entries.push(format!("{}/", name));
            } else {
                entries.push(name);
            }
        }
        entries.sort();
        Ok(entries)
    }

    /// Get file tree structure
    pub fn file_tree(&self, max_depth: usize) -> Result<FileTree> {
        self.build_tree(&self.local_path, 0, max_depth)
    }

    /// Find files by glob pattern
    pub fn find_files(&self, pattern: &str) -> Vec<String> {
        self.index.files()
            .iter()
            .filter(|f| {
                let path_str = f.path.to_string_lossy();
                if let Ok(glob) = globset::Glob::new(pattern) {
                    let matcher = glob.compile_matcher();
                    matcher.is_match(&f.path)
                } else {
                    path_str.contains(pattern)
                }
            })
            .map(|f| f.path.strip_prefix(&self.local_path)
                .unwrap_or(&f.path)
                .to_string_lossy()
                .to_string())
            .collect()
    }

    /// Get files by extension
    pub fn files_by_extension(&self, ext: &str) -> Vec<String> {
        self.index.files_by_ext(ext)
            .iter()
            .map(|f| f.path.strip_prefix(&self.local_path)
                .unwrap_or(&f.path)
                .to_string_lossy()
                .to_string())
            .collect()
    }

    // --- Private helpers ---

    fn extract_repo_name(url: &str) -> Result<String> {
        // Handle various GitHub URL formats
        let url = url.trim_end_matches('/').trim_end_matches(".git");

        let name = url.split('/')
            .last()
            .ok_or_else(|| GrepError::IndexError("Invalid repo URL".into()))?;

        // Include owner for uniqueness
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() >= 2 {
            let owner = parts[parts.len() - 2];
            Ok(format!("{}_{}", owner, name))
        } else {
            Ok(name.to_string())
        }
    }

    fn cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("emergentdb")
            .join("repos")
    }

    fn git_clone(url: &str, target: &Path) -> Result<()> {
        let status = Command::new("git")
            .args(["clone", "--depth", "1", url])
            .arg(target)
            .status()
            .map_err(|e| GrepError::Io(e))?;

        if !status.success() {
            return Err(GrepError::IndexError(format!("git clone failed for {}", url)));
        }
        Ok(())
    }

    fn git_pull(path: &Path) -> Result<()> {
        let status = Command::new("git")
            .args(["pull", "--ff-only"])
            .current_dir(path)
            .status()
            .map_err(|e| GrepError::Io(e))?;

        if !status.success() {
            // Non-fatal, repo might be fine
            eprintln!("Warning: git pull failed, using existing version");
        }
        Ok(())
    }

    fn build_tree(&self, path: &Path, depth: usize, max_depth: usize) -> Result<FileTree> {
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".")
            .to_string();

        let is_dir = path.is_dir();
        let mut children = Vec::new();

        if is_dir && depth < max_depth {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    let entry_name = entry_path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");

                    // Skip hidden and common ignored directories
                    if entry_name.starts_with('.') ||
                       entry_name == "node_modules" ||
                       entry_name == "target" ||
                       entry_name == "__pycache__" {
                        continue;
                    }

                    if let Ok(child) = self.build_tree(&entry_path, depth + 1, max_depth) {
                        children.push(child);
                    }
                }
            }
            children.sort_by(|a, b| {
                match (a.is_dir, b.is_dir) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.path.cmp(&b.path),
                }
            });
        }

        let metadata = fs::metadata(path).ok();

        Ok(FileTree {
            path: name,
            is_dir,
            children,
            size: metadata.as_ref().map(|m| m.len()),
            extension: if is_dir { None } else {
                path.extension().and_then(|e| e.to_str()).map(String::from)
            },
        })
    }
}

/// Quick function to explore a GitHub repo
pub fn explore(repo_url: &str) -> Result<RepoExplorer> {
    let mut explorer = RepoExplorer::from_github(repo_url)?;
    explorer.build_index()?;
    Ok(explorer)
}
