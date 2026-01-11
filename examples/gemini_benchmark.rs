//! EmergentDB Benchmark with Real Gemini Embeddings
//!
//! This benchmark uses actual embeddings from Google's Gemini API
//! to test index performance with semantically meaningful vectors.
//!
//! Run with: cargo run --release --example gemini_benchmark -p vector-core

use std::fs;
use std::time::Instant;
use serde::Deserialize;
use vector_core::{
    Embedding, NodeId, SearchResult, VectorIndex,
    FlatIndex, EmergentIndex, EmergentConfig,
    PrecomputedElitesGrid, PrecomputedElite,
    IndexConfig, DistanceMetric,
};

#[derive(Deserialize)]
struct GeminiEmbeddings {
    model: String,
    dimension: usize,
    count: usize,
    embeddings: Vec<EmbeddingEntry>,
}

#[derive(Deserialize)]
struct EmbeddingEntry {
    id: usize,
    source: String,
    text: String,
    vector: Vec<f32>,
}

const K: usize = 10;
const N_QUERIES: usize = 50;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("  EmergentDB Benchmark with Real Gemini Embeddings");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");

    // Check for scale argument
    let args: Vec<String> = std::env::args().collect();
    let scale = args.get(1).and_then(|s| s.parse::<usize>().ok());

    // Load Gemini embeddings
    let embeddings_path = match scale {
        Some(1000) => "tests/gemini_embeddings_1000.json",
        Some(5000) => "tests/gemini_embeddings_5000.json",
        Some(10000) => "tests/gemini_embeddings_10000.json",
        _ => "tests/gemini_embeddings.json",
    };
    println!("\n  Loading embeddings from: {}", embeddings_path);

    let json_content = match fs::read_to_string(embeddings_path) {
        Ok(content) => content,
        Err(e) => {
            println!("  Error: Could not load embeddings file: {}", e);
            println!("  Run: cd tests && python3 gemini_embedding_benchmark.py");
            return;
        }
    };

    let data: GeminiEmbeddings = match serde_json::from_str(&json_content) {
        Ok(data) => data,
        Err(e) => {
            println!("  Error parsing JSON: {}", e);
            return;
        }
    };

    println!("  Model: {}", data.model);
    println!("  Dimension: {}", data.dimension);
    println!("  Vectors: {}", data.count);

    // Convert to our format
    let vectors: Vec<(NodeId, Embedding)> = data.embeddings
        .iter()
        .map(|e| {
            let mut emb = Embedding::new(e.vector.clone());
            emb.normalize();
            (NodeId::new(e.id as u64), emb)
        })
        .collect();

    // Compute ground truth
    println!("\n  Computing ground truth for {} queries...", N_QUERIES);
    let query_indices: Vec<usize> = (0..N_QUERIES.min(vectors.len())).collect();
    let ground_truths = compute_ground_truth(&vectors, &query_indices);

    // Run benchmark with Flat (baseline)
    println!("\n══════════════════════════════════════════════════════════════════════════════════════");
    println!("  BASELINE: Flat Index (Brute Force)");
    println!("══════════════════════════════════════════════════════════════════════════════════════");

    let (flat_latency, flat_recall) = benchmark_flat(&vectors, &query_indices, &ground_truths, data.dimension);
    println!("  Flat: {:.1}μs avg latency, {:.1}% recall", flat_latency, flat_recall * 100.0);

    // Test all precomputed configurations
    println!("\n══════════════════════════════════════════════════════════════════════════════════════");
    println!("  PRE-COMPUTED ELITES GRID (Industry-Standard Configurations)");
    println!("══════════════════════════════════════════════════════════════════════════════════════");

    let grid = PrecomputedElitesGrid::new();
    let matching = grid.get_for_scale(vectors.len());

    println!("\n  {} configurations match dataset size of {} vectors", matching.len(), vectors.len());
    println!("\n  Recommendations by priority:");
    for priority in &["speed", "balanced", "accuracy", "max"] {
        let elite = grid.recommend(vectors.len(), priority);
        println!("    {:>10}: {} (expected: {:.0}%)",
            priority, elite.description, elite.expected_recall * 100.0);
    }

    println!("\n  ┌──────────────────────────────────────────┬──────────────┬──────────────┬──────────┬──────────┐");
    println!("  │ Configuration                            │ Build (ms)   │ Search (μs)  │ Recall   │ Expected │");
    println!("  ├──────────────────────────────────────────┼──────────────┼──────────────┼──────────┼──────────┤");

    for elite in matching.iter() {
        benchmark_precomputed_elite(elite, &vectors, &query_indices, &ground_truths, data.dimension);
    }

    println!("  └──────────────────────────────────────────┴──────────────┴──────────────┴──────────┴──────────┘");

    // Semantic search demo
    println!("\n══════════════════════════════════════════════════════════════════════════════════════");
    println!("  SEMANTIC SEARCH DEMO");
    println!("══════════════════════════════════════════════════════════════════════════════════════");

    semantic_search_demo(&vectors, &data.embeddings, data.dimension);
}

fn compute_ground_truth(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
) -> Vec<Vec<NodeId>> {
    query_indices
        .iter()
        .map(|&qi| {
            let query = &vectors[qi].1;
            let mut scores: Vec<(NodeId, f32)> = vectors
                .iter()
                .map(|(id, emb)| {
                    let score = vector_core::cosine_similarity_simd(query.as_slice(), emb.as_slice());
                    (*id, score)
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.into_iter().take(K).map(|(id, _)| id).collect()
        })
        .collect()
}

fn compute_recall(ground_truth: &[NodeId], results: &[SearchResult]) -> f32 {
    let result_ids: std::collections::HashSet<_> = results.iter().map(|r| r.id).collect();
    let relevant = ground_truth.iter().filter(|id| result_ids.contains(id)).count();
    relevant as f32 / ground_truth.len().min(K) as f32
}

fn benchmark_flat(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
    dim: usize,
) -> (f64, f32) {
    let config = IndexConfig::flat(dim, DistanceMetric::Cosine);
    let mut index = FlatIndex::new(config);

    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }

    let mut search_times = Vec::new();
    let mut recalls = Vec::new();

    for (i, &qi) in query_indices.iter().enumerate() {
        let query = &vectors[qi].1;
        let start = Instant::now();
        let results = index.search(query, K).unwrap();
        search_times.push(start.elapsed().as_micros() as f64);
        recalls.push(compute_recall(&ground_truths[i], &results));
    }

    let avg_latency = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    (avg_latency, avg_recall)
}

fn benchmark_precomputed_elite(
    elite: &PrecomputedElite,
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
    dim: usize,
) {
    let config = EmergentConfig {
        dim,
        metric: DistanceMetric::Cosine,
        ..EmergentConfig::fast()
    };
    let mut index = EmergentIndex::new(config);

    // Insert vectors
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }

    // Apply configuration
    let build_start = Instant::now();
    if let Err(e) = index.apply_precomputed_elite(elite) {
        let desc = truncate_str(elite.description, 38);
        println!("  │ {:<40} │ FAILED: {:?}", desc, e);
        return;
    }
    let build_time = build_start.elapsed();

    // Search benchmark
    let mut search_times = Vec::new();
    let mut recalls = Vec::new();

    for (i, &qi) in query_indices.iter().enumerate() {
        let query = &vectors[qi].1;
        let start = Instant::now();
        let results = index.search(query, K).unwrap();
        search_times.push(start.elapsed().as_micros() as f64);
        recalls.push(compute_recall(&ground_truths[i], &results));
    }

    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    let desc = truncate_str(elite.description, 38);
    println!(
        "  │ {:<40} │ {:>10.1}   │ {:>10.1}   │ {:>6.1}%  │ {:>6.1}%  │",
        desc,
        build_time.as_millis() as f64,
        avg_search,
        avg_recall * 100.0,
        elite.expected_recall * 100.0
    );
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() > max_len {
        let truncated: String = s.chars().take(max_len - 3).collect();
        format!("{}...", truncated)
    } else {
        s.to_string()
    }
}

fn semantic_search_demo(
    vectors: &[(NodeId, Embedding)],
    entries: &[EmbeddingEntry],
    dim: usize,
) {
    let config = EmergentConfig {
        dim,
        metric: DistanceMetric::Cosine,
        ..EmergentConfig::fast()
    };
    let mut index = EmergentIndex::new(config);

    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }

    // Use the recommended configuration
    let grid = PrecomputedElitesGrid::new();
    let elite = grid.recommend(vectors.len(), "balanced");
    println!("\n  Using: {} (expected recall: {:.0}%)", elite.description, elite.expected_recall * 100.0);

    index.apply_precomputed_elite(elite).unwrap();

    // Search for different topics
    let test_queries = [0, 10, 50, 100];

    for &qi in &test_queries {
        if qi >= entries.len() {
            continue;
        }

        let query_text = &entries[qi].text;
        let query_source = &entries[qi].source;
        let query = &vectors[qi].1;

        println!("\n  Query (from {}): \"{}\"", query_source, truncate_str(query_text, 50));

        let results = index.search(query, 5).unwrap();

        for (rank, result) in results.iter().enumerate() {
            let idx = result.id.0 as usize;
            if idx < entries.len() {
                let entry = &entries[idx];
                println!("    {}. [{:.4}] {} - \"{}\"",
                    rank + 1,
                    result.score,
                    entry.source,
                    truncate_str(&entry.text, 40)
                );
            }
        }
    }
}
