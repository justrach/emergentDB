//! Comprehensive RAG Benchmark: EmergentDB SIMD vs Evolved
//!
//! Tests both FlatIndex (SIMD brute-force) and EmergentIndex (MAP-Elites evolved)
//! directly in Rust without HTTP overhead.
//!
//! Usage: cargo run --release --example comprehensive_rag_benchmark <embeddings.json> <queries.json>

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use vector_core::{
    index::{
        flat::FlatIndex,
        emergent::{EmergentConfig, EmergentIndex},
        IndexConfig, VectorIndex,
    },
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
struct ConfigResult {
    name: String,
    description: String,
    index_type: String,
    results: Vec<SearchResult>,
    avg_latency_us: f64,
    build_time_ms: f64,
}

#[derive(Serialize)]
struct BenchmarkOutput {
    num_vectors: usize,
    num_queries: usize,
    dimension: usize,
    configs: Vec<ConfigResult>,
}

fn run_searches(
    index: &dyn VectorIndex,
    queries: &QueryData,
    chunks: &[String],
    k: usize,
) -> (Vec<SearchResult>, f64) {
    let mut results = Vec::new();
    let mut total_latency = 0.0;

    for (qi, query_emb) in queries.queries.iter().enumerate() {
        let query = Embedding::new(query_emb.clone());

        // Warmup on first query
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
            .map(|&i| chunks.get(i).cloned().unwrap_or_default())
            .collect();

        results.push(SearchResult {
            query_idx: qi,
            query_text: queries.query_texts.get(qi).cloned().unwrap_or_default(),
            result_indices,
            result_chunks,
            latency_us,
        });
    }

    let avg_latency = total_latency / results.len() as f64;
    (results, avg_latency)
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
    let k = 5;
    eprintln!(
        "Loaded {} embeddings, {} queries, {}D",
        data.embeddings.len(),
        queries.queries.len(),
        dim
    );

    let mut all_configs: Vec<ConfigResult> = Vec::new();

    // ==========================================================================
    // Test 1: SIMD Flat Index (Brute Force)
    // ==========================================================================
    eprintln!("\n--- Testing SIMD Flat Index ---");
    let build_start = Instant::now();

    let config = IndexConfig {
        dim,
        metric: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    };

    let mut flat_index = FlatIndex::new(config);
    for (i, emb) in data.embeddings.iter().enumerate() {
        flat_index
            .insert(NodeId(i as u64), Embedding::new(emb.clone()))
            .expect("Insert failed");
    }
    let flat_build_time = build_start.elapsed().as_millis() as f64;
    eprintln!("  Built in {:.1}ms", flat_build_time);

    let (flat_results, flat_avg) = run_searches(&flat_index, &queries, &data.chunks, k);
    eprintln!("  Avg latency: {:.1}us", flat_avg);

    all_configs.push(ConfigResult {
        name: "EmergentDB (SIMD Flat)".to_string(),
        description: "Brute-force with ARM NEON/AVX2 SIMD optimization".to_string(),
        index_type: "Flat".to_string(),
        results: flat_results,
        avg_latency_us: flat_avg,
        build_time_ms: flat_build_time,
    });

    // ==========================================================================
    // Test 2: EmergentIndex with MAP-Elites Evolution
    // ==========================================================================
    eprintln!("\n--- Testing EmergentIndex (MAP-Elites) ---");

    let emergent_config = EmergentConfig {
        dim,
        metric: DistanceMetric::Cosine,
        grid_size: 5,
        generations: 8,
        population_size: 6,
        eval_sample_size: data.embeddings.len().min(200),
        benchmark_queries: 20,
        early_stop_fitness: 0.95,
        min_generations: 3,
        ..Default::default()
    };

    let mut emergent_index = EmergentIndex::new(emergent_config);

    // Insert all vectors
    let insert_start = Instant::now();
    for (i, emb) in data.embeddings.iter().enumerate() {
        emergent_index
            .insert(NodeId(i as u64), Embedding::new(emb.clone()))
            .expect("Insert failed");
    }
    let insert_time = insert_start.elapsed().as_millis() as f64;
    eprintln!("  Inserted {} vectors in {:.1}ms", data.embeddings.len(), insert_time);

    // Run evolution
    eprintln!("  Running MAP-Elites evolution...");
    let evolve_start = Instant::now();
    let evolution_result = emergent_index.evolve();
    let evolve_time = evolve_start.elapsed().as_millis() as f64;

    let (evolved_index_type, evolved_description) = match &evolution_result {
        Ok(elite) => {
            let idx_type = format!("{}", elite.genome.index_type);
            let desc = format!(
                "MAP-Elites evolved to {} (fitness: {:.3}, recall: {:.1}%)",
                elite.genome.index_type,
                elite.fitness,
                elite.metrics.recall_at_10 * 100.0
            );
            eprintln!("  Evolved to: {} in {:.1}ms", elite.genome.index_type, evolve_time);
            eprintln!("    Fitness: {:.3}", elite.fitness);
            eprintln!("    Recall: {:.1}%", elite.metrics.recall_at_10 * 100.0);
            eprintln!("    Latency: {:.1}us", elite.metrics.query_latency_us);
            (idx_type, desc)
        }
        Err(e) => {
            eprintln!("  Evolution failed: {} - using default", e);
            ("Flat".to_string(), "Evolution failed, using Flat fallback".to_string())
        }
    };

    let total_build_time = insert_time + evolve_time;

    let (evolved_results, evolved_avg) = run_searches(&emergent_index, &queries, &data.chunks, k);
    eprintln!("  Avg latency: {:.1}us", evolved_avg);

    all_configs.push(ConfigResult {
        name: "EmergentDB (Evolved)".to_string(),
        description: evolved_description,
        index_type: evolved_index_type,
        results: evolved_results,
        avg_latency_us: evolved_avg,
        build_time_ms: total_build_time,
    });

    // ==========================================================================
    // Test 3: HNSW with default params (for comparison)
    // ==========================================================================
    eprintln!("\n--- Testing HNSW Index (M=16, ef=50) ---");
    use vector_core::index::hnsw::HnswIndex;

    let hnsw_build_start = Instant::now();
    let hnsw_config = IndexConfig {
        dim,
        metric: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    };

    let mut hnsw_index = HnswIndex::new(hnsw_config);
    for (i, emb) in data.embeddings.iter().enumerate() {
        hnsw_index
            .insert(NodeId(i as u64), Embedding::new(emb.clone()))
            .expect("Insert failed");
    }
    let hnsw_build_time = hnsw_build_start.elapsed().as_millis() as f64;
    eprintln!("  Built in {:.1}ms", hnsw_build_time);

    let (hnsw_results, hnsw_avg) = run_searches(&hnsw_index, &queries, &data.chunks, k);
    eprintln!("  Avg latency: {:.1}us", hnsw_avg);

    all_configs.push(ConfigResult {
        name: "EmergentDB (HNSW Default)".to_string(),
        description: "HNSW with M=16, ef_construction=200, ef_search=50".to_string(),
        index_type: "HNSW".to_string(),
        results: hnsw_results,
        avg_latency_us: hnsw_avg,
        build_time_ms: hnsw_build_time,
    });

    // ==========================================================================
    // Test 4: HNSW with high-recall params
    // ==========================================================================
    eprintln!("\n--- Testing HNSW Index (M=32, ef=100) ---");

    let hnsw_hr_build_start = Instant::now();
    let hnsw_hr_config = IndexConfig {
        dim,
        metric: DistanceMetric::Cosine,
        m: 32,
        ef_construction: 300,
        ef_search: 100,
    };

    let mut hnsw_hr_index = HnswIndex::new(hnsw_hr_config);
    for (i, emb) in data.embeddings.iter().enumerate() {
        hnsw_hr_index
            .insert(NodeId(i as u64), Embedding::new(emb.clone()))
            .expect("Insert failed");
    }
    let hnsw_hr_build_time = hnsw_hr_build_start.elapsed().as_millis() as f64;
    eprintln!("  Built in {:.1}ms", hnsw_hr_build_time);

    let (hnsw_hr_results, hnsw_hr_avg) = run_searches(&hnsw_hr_index, &queries, &data.chunks, k);
    eprintln!("  Avg latency: {:.1}us", hnsw_hr_avg);

    all_configs.push(ConfigResult {
        name: "EmergentDB (HNSW High-Recall)".to_string(),
        description: "HNSW with M=32, ef_construction=300, ef_search=100".to_string(),
        index_type: "HNSW".to_string(),
        results: hnsw_hr_results,
        avg_latency_us: hnsw_hr_avg,
        build_time_ms: hnsw_hr_build_time,
    });

    // ==========================================================================
    // Summary
    // ==========================================================================
    eprintln!("\n========================================");
    eprintln!("SUMMARY (sorted by latency)");
    eprintln!("========================================");

    all_configs.sort_by(|a, b| a.avg_latency_us.partial_cmp(&b.avg_latency_us).unwrap());

    for cfg in &all_configs {
        eprintln!(
            "  {:30} {:>8.1}us  (build: {:>6.1}ms)",
            cfg.name, cfg.avg_latency_us, cfg.build_time_ms
        );
    }

    // Output JSON
    let output = BenchmarkOutput {
        num_vectors: data.embeddings.len(),
        num_queries: queries.queries.len(),
        dimension: dim,
        configs: all_configs,
    };

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
