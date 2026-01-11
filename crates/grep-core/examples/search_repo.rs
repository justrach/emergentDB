//! Example: Use grep-core to search a cloned GitHub repo

use grep_core::{RepoExplorer, SearchQuery};

fn main() -> grep_core::Result<()> {
    // Load the already-cloned python-genai repo
    // Note: We cloned to ~/.cache manually, not the macOS Library/Caches
    let home = std::env::var("HOME").unwrap();
    let mut explorer = RepoExplorer::from_local(
        std::path::Path::new(&home)
            .join(".cache/emergentdb/repos/googleapis_python-genai")
    )?;

    println!("📦 Repo: {}", explorer.info().name);
    println!("📂 Path: {}", explorer.info().local_path);

    // Build the search index
    print!("🔨 Building index... ");
    let file_count = explorer.build_index()?;
    println!("indexed {} files", file_count);

    // Search for embeddings
    println!("\n🔍 Searching for 'embed' patterns...\n");

    let results = explorer.search_advanced(&SearchQuery::regex(r"embed.*content|embedding")
        .case_insensitive()
        .limit(20)
        .context(1))?;

    for result in &results {
        let rel_path = result.file.strip_prefix(&explorer.local_path)
            .unwrap_or(&result.file);
        println!("📄 {} ({} matches, score: {:.2})",
            rel_path.display(),
            result.matches.len(),
            result.score);

        for m in result.matches.iter().take(3) {
            println!("   L{}: {}", m.line_number, m.line_content.trim());
        }
        println!();
    }

    // Also find all files with "embed" in the name
    println!("📁 Files with 'embed' in name:");
    for file in explorer.find_files("*embed*") {
        println!("   {}", file);
    }

    Ok(())
}
