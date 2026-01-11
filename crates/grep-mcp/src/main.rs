//! grep-mcp: MCP server for searching GitHub repositories
//!
//! Exposes grep-core functionality as MCP tools for LLM agents.

use std::sync::Arc;
use tokio::sync::RwLock;
use rmcp::{
    ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{ServerCapabilities, ServerInfo, CallToolResult, Content},
    tool, tool_handler, tool_router,
    transport::stdio,
    ErrorData as McpError,
    ServiceExt,
    schemars,
};
use tracing_subscriber::{self, EnvFilter};
use serde::Deserialize;

use grep_core::{RepoExplorer, SearchQuery};

// ============================================================================
// Tool Parameter Types
// ============================================================================

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CloneRepoRequest {
    #[schemars(description = "GitHub repository URL (e.g., https://github.com/user/repo)")]
    pub url: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(description = "Search pattern (literal text or regex)")]
    pub pattern: String,
    #[schemars(description = "Use regex matching (default: false)")]
    #[serde(default)]
    pub regex: bool,
    #[schemars(description = "Case insensitive search (default: false)")]
    #[serde(default)]
    pub case_insensitive: bool,
    #[schemars(description = "Maximum number of results (default: 20)")]
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[schemars(description = "File glob pattern to filter (e.g., *.rs, **/*.py)")]
    pub file_pattern: Option<String>,
}

fn default_limit() -> usize { 20 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReadFileRequest {
    #[schemars(description = "Relative path to the file within the repo")]
    pub path: String,
    #[schemars(description = "Start line (1-indexed, optional)")]
    pub start_line: Option<usize>,
    #[schemars(description = "End line (1-indexed, optional)")]
    pub end_line: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListDirRequest {
    #[schemars(description = "Relative path to the directory (use '.' for root)")]
    #[serde(default = "default_dir")]
    pub path: String,
}

fn default_dir() -> String { ".".to_string() }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FileTreeRequest {
    #[schemars(description = "Maximum depth to traverse (default: 3)")]
    #[serde(default = "default_depth")]
    pub depth: usize,
}

fn default_depth() -> usize { 3 }

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FindFilesRequest {
    #[schemars(description = "Glob pattern to match files (e.g., **/*.rs, src/*.py)")]
    pub pattern: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct FilesByExtRequest {
    #[schemars(description = "File extension without dot (e.g., rs, py, ts)")]
    pub extension: String,
}

// ============================================================================
// MCP Server
// ============================================================================

#[derive(Clone)]
pub struct GrepServer {
    explorer: Arc<RwLock<Option<RepoExplorer>>>,
    tool_router: ToolRouter<Self>,
}

impl GrepServer {
    pub fn new() -> Self {
        Self {
            explorer: Arc::new(RwLock::new(None)),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl GrepServer {
    #[tool(description = "Clone a GitHub repository and prepare it for searching. Call this first before using other tools.")]
    async fn clone_repo(&self, Parameters(req): Parameters<CloneRepoRequest>) -> Result<CallToolResult, McpError> {
        let result = tokio::task::spawn_blocking(move || {
            grep_core::RepoExplorer::from_github(&req.url)
        }).await.map_err(|e| McpError::internal_error(format!("Task failed: {}", e), None))?;

        match result {
            Ok(mut explorer) => {
                // Build the index
                let index_result = explorer.build_index();
                let info = explorer.info();

                // Store the explorer
                let mut guard = self.explorer.write().await;
                *guard = Some(explorer);

                match index_result {
                    Ok(file_count) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Repository cloned and indexed successfully!\n\nRepo: {}\nPath: {}\nFiles indexed: {}\n\nYou can now use search, read_file, list_dir, file_tree, find_files, and files_by_ext tools.",
                        info.name, info.local_path, file_count
                    ))])),
                    Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Repository cloned but indexing failed: {}\n\nPath: {}",
                        e, info.local_path
                    ))]))
                }
            }
            Err(e) => Err(McpError::internal_error(format!("Failed to clone repo: {}", e), None))
        }
    }

    #[tool(description = "Search for patterns in the cloned repository. Supports literal and regex patterns.")]
    async fn search(&self, Parameters(req): Parameters<SearchRequest>) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        let explorer = guard.as_ref()
            .ok_or_else(|| McpError::invalid_request("No repository loaded. Call clone_repo first.", None))?;

        let mut query = if req.regex {
            SearchQuery::regex(&req.pattern)
        } else {
            SearchQuery::literal(&req.pattern)
        };

        if req.case_insensitive {
            query = query.case_insensitive();
        }
        query = query.limit(req.limit).context(2);

        if let Some(ref fp) = req.file_pattern {
            query = query.in_files(fp);
        }

        match explorer.search_advanced(&query) {
            Ok(results) => {
                if results.is_empty() {
                    return Ok(CallToolResult::success(vec![Content::text("No matches found.")]));
                }

                let mut output = format!("Found matches in {} files:\n\n", results.len());
                for result in &results {
                    let rel_path = result.file.strip_prefix(&explorer.local_path)
                        .unwrap_or(&result.file);
                    output.push_str(&format!("## {} ({} matches)\n", rel_path.display(), result.matches.len()));

                    for m in result.matches.iter().take(5) {
                        output.push_str(&format!("L{}: {}\n", m.line_number, m.line_content.trim()));
                    }
                    if result.matches.len() > 5 {
                        output.push_str(&format!("... and {} more matches\n", result.matches.len() - 5));
                    }
                    output.push('\n');
                }
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError::internal_error(format!("Search failed: {}", e), None))
        }
    }

    #[tool(description = "Read a file from the cloned repository. Can read specific line ranges.")]
    async fn read_file(&self, Parameters(req): Parameters<ReadFileRequest>) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        let explorer = guard.as_ref()
            .ok_or_else(|| McpError::invalid_request("No repository loaded. Call clone_repo first.", None))?;

        let content = match (req.start_line, req.end_line) {
            (Some(start), Some(end)) => {
                explorer.read_file_lines(&req.path, start, end)
                    .map(|lines| lines.join("\n"))
            }
            _ => explorer.read_file(&req.path)
        };

        match content {
            Ok(text) => Ok(CallToolResult::success(vec![Content::text(text)])),
            Err(e) => Err(McpError::internal_error(format!("Failed to read file: {}", e), None))
        }
    }

    #[tool(description = "List files and directories in a path within the cloned repository.")]
    async fn list_dir(&self, Parameters(req): Parameters<ListDirRequest>) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        let explorer = guard.as_ref()
            .ok_or_else(|| McpError::invalid_request("No repository loaded. Call clone_repo first.", None))?;

        match explorer.list_dir(&req.path) {
            Ok(entries) => {
                let output = entries.join("\n");
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError::internal_error(format!("Failed to list directory: {}", e), None))
        }
    }

    #[tool(description = "Get a tree view of the repository structure up to a specified depth.")]
    async fn file_tree(&self, Parameters(req): Parameters<FileTreeRequest>) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        let explorer = guard.as_ref()
            .ok_or_else(|| McpError::invalid_request("No repository loaded. Call clone_repo first.", None))?;

        match explorer.file_tree(req.depth) {
            Ok(tree) => {
                let output = format_tree(&tree, 0);
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Err(McpError::internal_error(format!("Failed to get file tree: {}", e), None))
        }
    }

    #[tool(description = "Find files matching a glob pattern (e.g., **/*.rs, src/**/*.py).")]
    async fn find_files(&self, Parameters(req): Parameters<FindFilesRequest>) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        let explorer = guard.as_ref()
            .ok_or_else(|| McpError::invalid_request("No repository loaded. Call clone_repo first.", None))?;

        let files = explorer.find_files(&req.pattern);
        if files.is_empty() {
            Ok(CallToolResult::success(vec![Content::text("No files found matching the pattern.")]))
        } else {
            let output = format!("Found {} files:\n\n{}", files.len(), files.join("\n"));
            Ok(CallToolResult::success(vec![Content::text(output)]))
        }
    }

    #[tool(description = "Get all files with a specific extension (e.g., rs, py, ts).")]
    async fn files_by_ext(&self, Parameters(req): Parameters<FilesByExtRequest>) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        let explorer = guard.as_ref()
            .ok_or_else(|| McpError::invalid_request("No repository loaded. Call clone_repo first.", None))?;

        let files = explorer.files_by_extension(&req.extension);
        if files.is_empty() {
            Ok(CallToolResult::success(vec![Content::text(format!("No .{} files found.", req.extension))]))
        } else {
            let output = format!("Found {} .{} files:\n\n{}", files.len(), req.extension, files.join("\n"));
            Ok(CallToolResult::success(vec![Content::text(output)]))
        }
    }

    #[tool(description = "Get information about the currently loaded repository.")]
    async fn repo_info(&self) -> Result<CallToolResult, McpError> {
        let guard = self.explorer.read().await;
        match guard.as_ref() {
            Some(explorer) => {
                let info = explorer.info();
                let output = format!(
                    "Repository: {}\nURL: {}\nLocal path: {}\nFiles: {}\nIndexed: {}",
                    info.name, info.url, info.local_path, info.file_count, info.indexed
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            None => Ok(CallToolResult::success(vec![Content::text("No repository loaded. Call clone_repo first.")]))
        }
    }
}

#[tool_handler]
impl ServerHandler for GrepServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "GitHub Repository Search Server - Search and explore any GitHub repository.\n\n\
                 Tools available:\n\
                 - clone_repo: Clone a GitHub repo (call this first!)\n\
                 - search: Search for patterns in code\n\
                 - read_file: Read file contents\n\
                 - list_dir: List directory contents\n\
                 - file_tree: Get repository structure\n\
                 - find_files: Find files by glob pattern\n\
                 - files_by_ext: Find files by extension\n\
                 - repo_info: Get info about loaded repo".into()
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// Helper to format file tree
fn format_tree(tree: &grep_core::FileTree, indent: usize) -> String {
    let mut output = String::new();
    let prefix = "  ".repeat(indent);

    if tree.is_dir {
        output.push_str(&format!("{}{}/\n", prefix, tree.path));
        for child in &tree.children {
            output.push_str(&format_tree(child, indent + 1));
        }
    } else {
        let size_str = tree.size.map(|s| format!(" ({}B)", s)).unwrap_or_default();
        output.push_str(&format!("{}{}{}\n", prefix, tree.path, size_str));
    }
    output
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging to stderr (stdout is for MCP protocol)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting grep-mcp server");

    let service = GrepServer::new().serve(stdio()).await.inspect_err(|e| {
        tracing::error!("Server error: {:?}", e);
    })?;

    service.waiting().await?;
    Ok(())
}
