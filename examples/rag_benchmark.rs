//! RAG Benchmark: EmergentDB SIMD search on real Gemini embeddings
//!
//! This binary reads embeddings from a JSON file and runs searches,
//! outputting results that can be compared with ChromaDB.
//!
//! Usage: cargo run --release --example rag_benchmark <embeddings.json> <queries.json>

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use vector_core::{
    index::{flat::FlatIndex, IndexConfig, VectorIndex},
    DistanceMetric, Embedding, NodeId,
};

#[derive(Deserialize)]
struct EmbeddingData {
    embeddings: Vec<Vec<f32>>,
    chunks: Vec<String>,
}

#[derive(Deserialize)]
struct QueryData {
    queries: Vec<Vec<f32>>,
    query_texts: Vec<String>,
}

#[derive(Serialize)]
struct SearchResult {
    query_idx: usize,
    query_text: String,
    result_indices: Vec<usize>,
    result_chunks: Vec<String>,
    latency_us: f64,
}

#[derive(Serialize)]
struct BenchmarkOutput {
    database: String,
    num_vectors: usize,
    num_queries: usize,
    dimension: usize,
    results: Vec<SearchResult>,
    avg_latency_us: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <embeddings.json> <queries.json>", args[0]);
        std::process::exit(1);
    }

    let embeddings_path = &args[1];
    let queries_path = &args[2];

    // Load data
    eprintln!("Loading embeddings from {}...", embeddings_path);
    let data: EmbeddingData = serde_json::from_str(
        &fs::read_to_string(embeddings_path).expect("Failed to read embeddings"),
    )
    .expect("Failed to parse embeddings JSON");

    eprintln!("Loading queries from {}...", queries_path);
    let queries: QueryData =
        serde_json::from_str(&fs::read_to_string(queries_path).expect("Failed to read queries"))
            .expect("Failed to parse queries JSON");

    let dim = data.embeddings[0].len();
    eprintln!(
        "Loaded {} embeddings, {} queries, {}D",
        data.embeddings.len(),
        queries.queries.len(),
        dim
    );

    // Build index with proper config
    eprintln!("Building Flat index (SIMD optimized)...");
    let build_start = Instant::now();

    let config = IndexConfig {
        dim,
        metric: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    };

    let mut index = FlatIndex::new(config);

    for (i, emb) in data.embeddings.iter().enumerate() {
        index
            .insert(NodeId(i as u64), Embedding::new(emb.clone()))
            .expect("Insert failed");
    }
    let build_time = build_start.elapsed();
    eprintln!("Index built in {:?}", build_time);

    // Run searches
    eprintln!("Running {} searches...", queries.queries.len());
    let k = 5;
    let mut results = Vec::new();
    let mut total_latency = 0.0;

    for (qi, query_emb) in queries.queries.iter().enumerate() {
        let query = Embedding::new(query_emb.clone());

        // Warmup
        if qi == 0 {
            for _ in 0..10 {
                let _ = index.search(&query, k);
            }
        }

        // Timed search
        let start = Instant::now();
        let search_results = index.search(&query, k).expect("Search failed");
        let latency_us = start.elapsed().as_nanos() as f64 / 1000.0;
        total_latency += latency_us;

        let result_indices: Vec<usize> = search_results.iter().map(|r| r.id.0 as usize).collect();
        let result_chunks: Vec<String> = result_indices
            .iter()
            .map(|&i| data.chunks.get(i).cloned().unwrap_or_default())
            .collect();

        results.push(SearchResult {
            query_idx: qi,
            query_text: queries.query_texts.get(qi).cloned().unwrap_or_default(),
            result_indices,
            result_chunks,
            latency_us,
        });

        eprintln!("  Query {}: {:.1}us", qi + 1, latency_us);
    }

    let avg_latency = total_latency / results.len() as f64;
    eprintln!("\nAverage latency: {:.1}us", avg_latency);

    // Output JSON
    let output = BenchmarkOutput {
        database: "EmergentDB (SIMD Flat)".to_string(),
        num_vectors: data.embeddings.len(),
        num_queries: queries.queries.len(),
        dimension: dim,
        results,
        avg_latency_us: avg_latency,
    };

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
