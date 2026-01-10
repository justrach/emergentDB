//! Scaled EmergentDB Benchmark
//!
//! Tests performance at different scales:
//! - 1,000 vectors (small)
//! - 10,000 vectors (medium)
//! - 100,000 vectors (large)
//!
//! Simulates 100 document chunks with additional random vectors.

use std::time::Instant;
use vector_core::{
    index::{
        flat::FlatIndex,
        hnsw::HnswIndex,
        emergent::{EmergentConfig, EmergentIndex},
        IndexConfig, VectorIndex,
    },
    DistanceMetric, Embedding, NodeId,
};

/// Generate random unit vectors
fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            let mut vec: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            vec.iter_mut().for_each(|x| *x /= norm);
            vec
        })
        .collect()
}

/// Generate document-like vectors (clustered, simulating 100 chunks)
fn generate_document_vectors(n_chunks: usize, n_random: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(n_chunks + n_random);

    // Generate 100 "document chunk" vectors - these are clustered
    // Simulates embeddings from the same document having some similarity
    let base_doc: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
    let base_norm: f32 = base_doc.iter().map(|x| x * x).sum::<f32>().sqrt();
    let base_doc: Vec<f32> = base_doc.iter().map(|x| x / base_norm).collect();

    for chunk_id in 0..n_chunks {
        // Each chunk is base + noise (simulates related content)
        let noise_scale = 0.3 + (chunk_id as f32 / n_chunks as f32) * 0.4;
        let mut chunk: Vec<f32> = base_doc
            .iter()
            .map(|&x| x + (rng.r#gen::<f32>() - 0.5) * noise_scale)
            .collect();

        // Normalize
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        chunk.iter_mut().for_each(|x| *x /= norm);
        vectors.push(chunk);
    }

    // Add random vectors (other documents in the database)
    for _ in 0..n_random {
        let mut vec: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        vec.iter_mut().for_each(|x| *x /= norm);
        vectors.push(vec);
    }

    vectors
}

/// Compute recall@k
fn compute_recall(ground_truth: &[NodeId], results: &[NodeId], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
    let result_set: std::collections::HashSet<_> = results.iter().take(k).collect();
    let hits = gt_set.intersection(&result_set).count();
    hits as f32 / k as f32
}

fn benchmark_scale(n_vectors: usize, dim: usize, n_queries: usize) {
    println!("\n{}", "=".repeat(70));
    println!("SCALE TEST: {} vectors, {} dimensions", n_vectors, dim);
    println!("{}", "=".repeat(70));

    // Generate vectors: 100 document chunks + rest random
    let n_chunks = 100.min(n_vectors);
    let n_random = n_vectors.saturating_sub(100);

    println!("\n  Generating {} document chunks + {} random vectors...", n_chunks, n_random);
    let start = Instant::now();
    let vectors = generate_document_vectors(n_chunks, n_random, dim);
    println!("  Generated in {:.1}ms", start.elapsed().as_millis());

    // Use first N vectors as queries (document chunks)
    let query_indices: Vec<usize> = (0..n_queries.min(n_chunks)).collect();
    let k = 10;

    // =========================================================================
    // GROUND TRUTH (Flat Index)
    // =========================================================================
    println!("\n  [1/3] Building ground truth (Flat index)...");
    let flat_config = IndexConfig::flat(dim, DistanceMetric::Cosine);
    let mut flat_index = FlatIndex::new(flat_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        flat_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    let flat_insert_time = start.elapsed().as_millis();

    // Compute ground truth
    let mut ground_truths = Vec::new();
    let start = Instant::now();
    for &qi in &query_indices {
        let query = Embedding::new(vectors[qi].clone());
        let res = flat_index.search(&query, k).unwrap();
        ground_truths.push(res.iter().map(|r| r.id).collect::<Vec<_>>());
    }
    let flat_search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;
    println!("    Insert: {}ms, Search: {:.1}μs/query", flat_insert_time, flat_search_time);

    // =========================================================================
    // HNSW (Manual)
    // =========================================================================
    println!("\n  [2/3] Benchmarking HNSW (manual)...");
    let hnsw_config = IndexConfig::hnsw(dim, DistanceMetric::Cosine);
    let mut hnsw_index = HnswIndex::new(hnsw_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        hnsw_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    let hnsw_insert_time = start.elapsed().as_millis();

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {
        let query = Embedding::new(vectors[qi].clone());
        let res = hnsw_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }
    let hnsw_search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;
    let hnsw_recall = total_recall / query_indices.len() as f32;
    println!("    Insert: {}ms, Search: {:.1}μs/query, Recall@{}: {:.1}%",
             hnsw_insert_time, hnsw_search_time, k, hnsw_recall * 100.0);

    // =========================================================================
    // EMERGENT (Auto)
    // =========================================================================
    println!("\n  [3/3] Benchmarking EMERGENT (auto-selection)...");

    // Use search_first for optimal query performance
    let mut emergent_config = EmergentConfig::search_first();
    emergent_config.dim = dim;
    emergent_config.metric = DistanceMetric::Cosine;

    // Scale sample size with dataset - need representative sample for accurate evolution
    // Without this, evolution underestimates search cost at scale
    emergent_config.eval_sample_size = (n_vectors / 5).max(500).min(5000);
    emergent_config.benchmark_queries = (n_vectors / 50).max(50).min(500);

    if n_vectors >= 10000 {
        emergent_config.generations = 15;
    }
    if n_vectors >= 50000 {
        emergent_config.generations = 20;
    }

    let mut emergent_index = EmergentIndex::new(emergent_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        emergent_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    let insert_time = start.elapsed().as_millis();

    let evolve_start = Instant::now();
    let elite = emergent_index.evolve().unwrap();
    let evolve_time = evolve_start.elapsed().as_millis();

    println!("    Evolution: {}ms", evolve_time);
    println!("    Selected: {} (fitness: {:.3})", elite.genome.index_type, elite.fitness);
    println!("    Archive coverage: {:.1}%", emergent_index.archive_coverage());

    // InsertQD results
    if let Some(insert_elite) = emergent_index.get_best_insert_elite() {
        println!("    Insert strategy: {} ({:.1}M vec/s)",
                 insert_elite.genome.strategy,
                 insert_elite.metrics.throughput_vps / 1_000_000.0);
    }

    // Benchmark search
    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {
        let query = Embedding::new(vectors[qi].clone());
        let res = emergent_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }
    let emergent_search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;
    let emergent_recall = total_recall / query_indices.len() as f32;

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n  {}", "-".repeat(60));
    println!("  RESULTS @ {} vectors:", n_vectors);
    println!("  {}", "-".repeat(60));
    println!("  {:15} {:>12} {:>12} {:>10}", "Index", "Insert(ms)", "Search(μs)", "Recall@10");
    println!("  {:15} {:>12} {:>12.1} {:>9.1}%", "Flat", flat_insert_time, flat_search_time, 100.0);
    println!("  {:15} {:>12} {:>12.1} {:>9.1}%", "HNSW (Manual)", hnsw_insert_time, hnsw_search_time, hnsw_recall * 100.0);
    println!("  {:15} {:>12} {:>12.1} {:>9.1}%",
             format!("Emergent ({})", elite.genome.index_type),
             insert_time + evolve_time as u128,
             emergent_search_time,
             emergent_recall * 100.0);

    // Speedup calculations
    let speedup_vs_flat = flat_search_time / emergent_search_time;
    let speedup_vs_hnsw = hnsw_search_time / emergent_search_time;

    println!("\n  Speedup vs Flat: {:.1}x", speedup_vs_flat);
    println!("  Speedup vs HNSW: {:.1}x", speedup_vs_hnsw);
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("EmergentDB Scale Benchmark");
    println!("Testing with 100 document chunks + random vectors");
    println!("{}", "=".repeat(70));

    let dim = 768; // Standard embedding dimension
    let n_queries = 20;

    // Test at different scales
    benchmark_scale(1_000, dim, n_queries);
    benchmark_scale(10_000, dim, n_queries);
    benchmark_scale(50_000, dim, n_queries);

    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK COMPLETE");
    println!("{}", "=".repeat(70));
}
