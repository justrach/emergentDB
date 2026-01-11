//! Search functionality with regex and literal matching

use std::path::{Path, PathBuf};
use std::fs;
use regex::Regex;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::{Result, CodeIndex, FileEntry};
use crate::ranking::RankingConfig;

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub pattern: String,
    pub is_regex: bool,
    pub case_sensitive: bool,
    pub whole_word: bool,
    pub file_pattern: Option<String>,
    pub max_results: usize,
    pub context_lines: usize,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            pattern: String::new(),
            is_regex: false,
            case_sensitive: true,
            whole_word: false,
            file_pattern: None,
            max_results: 100,
            context_lines: 2,
        }
    }
}

impl SearchQuery {
    pub fn literal(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            ..Default::default()
        }
    }

    pub fn regex(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            is_regex: true,
            ..Default::default()
        }
    }

    pub fn case_insensitive(mut self) -> Self {
        self.case_sensitive = false;
        self
    }

    pub fn whole_word(mut self) -> Self {
        self.whole_word = true;
        self
    }

    pub fn in_files(mut self, pattern: &str) -> Self {
        self.file_pattern = Some(pattern.to_string());
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    pub fn context(mut self, lines: usize) -> Self {
        self.context_lines = lines;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub line_number: usize,
    pub line_content: String,
    pub match_start: usize,
    pub match_end: usize,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file: PathBuf,
    pub matches: Vec<Match>,
    pub score: f32,
}

pub struct Searcher {
    pub index: CodeIndex,
    ranking: RankingConfig,
}

impl Searcher {
    pub fn new(index: CodeIndex) -> Self {
        Self {
            index,
            ranking: RankingConfig::default(),
        }
    }

    pub fn with_ranking(mut self, config: RankingConfig) -> Self {
        self.ranking = config;
        self
    }

    /// Search across all indexed files
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let regex = self.build_regex(query)?;
        let files = self.filter_files(query);

        let mut results: Vec<SearchResult> = files
            .par_iter()
            .filter_map(|file| {
                self.search_file(&file.path, &regex, query).ok().flatten()
            })
            .collect();

        // Rank results
        self.rank_results(&mut results, query);

        // Limit results
        results.truncate(query.max_results);

        Ok(results)
    }

    fn build_regex(&self, query: &SearchQuery) -> Result<Regex> {
        let mut pattern = if query.is_regex {
            query.pattern.clone()
        } else {
            regex::escape(&query.pattern)
        };

        if query.whole_word {
            pattern = format!(r"\b{}\b", pattern);
        }

        let regex = if query.case_sensitive {
            Regex::new(&pattern)?
        } else {
            Regex::new(&format!("(?i){}", pattern))?
        };

        Ok(regex)
    }

    fn filter_files(&self, query: &SearchQuery) -> Vec<FileEntry> {
        let files = self.index.files();

        if let Some(ref pattern) = query.file_pattern {
            if let Ok(glob) = globset::Glob::new(pattern) {
                let matcher = glob.compile_matcher();
                return files.into_iter()
                    .filter(|f| matcher.is_match(&f.path))
                    .collect();
            }
        }

        files
    }

    fn search_file(&self, path: &Path, regex: &Regex, query: &SearchQuery) -> Result<Option<SearchResult>> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        let mut matches = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            for m in regex.find_iter(line) {
                let context_before: Vec<String> = lines
                    .iter()
                    .skip(i.saturating_sub(query.context_lines))
                    .take(query.context_lines.min(i))
                    .map(|s| s.to_string())
                    .collect();

                let context_after: Vec<String> = lines
                    .iter()
                    .skip(i + 1)
                    .take(query.context_lines)
                    .map(|s| s.to_string())
                    .collect();

                matches.push(Match {
                    line_number: i + 1,
                    line_content: line.to_string(),
                    match_start: m.start(),
                    match_end: m.end(),
                    context_before,
                    context_after,
                });
            }
        }

        if matches.is_empty() {
            return Ok(None);
        }

        Ok(Some(SearchResult {
            file: path.to_path_buf(),
            matches,
            score: 0.0, // Will be set by ranking
        }))
    }

    fn rank_results(&self, results: &mut Vec<SearchResult>, _query: &SearchQuery) {
        for result in results.iter_mut() {
            let mut score = 0.0;

            // More matches = higher score
            score += (result.matches.len() as f32) * self.ranking.match_count_weight;

            // Prefer shorter files (more focused)
            let file_entry = self.index.get_file(&result.file);
            if let Some(entry) = file_entry {
                let density = result.matches.len() as f32 / entry.line_count.max(1) as f32;
                score += density * self.ranking.density_weight;

                // Prefer recently modified files
                score += (entry.modified as f32 / 1e10) * self.ranking.recency_weight;
            }

            // Prefer certain file types
            if let Some(ext) = result.file.extension().and_then(|e| e.to_str()) {
                if self.ranking.priority_extensions.contains(&ext.to_string()) {
                    score += self.ranking.extension_boost;
                }
            }

            result.score = score;
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }
}

/// Quick search function for one-off searches
pub fn search<P: AsRef<Path>>(root: P, pattern: &str) -> Result<Vec<SearchResult>> {
    let index = CodeIndex::from_path(root);
    index.build()?;
    let searcher = Searcher::new(index);
    searcher.search(&SearchQuery::literal(pattern))
}

/// Quick regex search
pub fn search_regex<P: AsRef<Path>>(root: P, pattern: &str) -> Result<Vec<SearchResult>> {
    let index = CodeIndex::from_path(root);
    index.build()?;
    let searcher = Searcher::new(index);
    searcher.search(&SearchQuery::regex(pattern))
}
