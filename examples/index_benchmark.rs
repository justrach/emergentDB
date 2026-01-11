//! EmergentDB Index Benchmark
//!
//! Directly benchmarks all index types without HTTP overhead.
//! Run with: cargo run --release --example index_benchmark -p vector-core
//!
//! This will show:
//! - Insert throughput for each index type
//! - Index build time
//! - Search latency
//! - Recall@10 accuracy

use std::time::Instant;
use rand::Rng;
use vector_core::{
    Embedding, NodeId, SearchResult, VectorIndex,
    FlatIndex, HnswIndex, IvfIndex,
    EmergentIndex, EmergentConfig, IndexPreset,
    PrecomputedElitesGrid, PrecomputedElite,
    IndexConfig, DistanceMetric,
    index::ivf::IvfConfig,
};

// Benchmark configuration
const VECTOR_DIM: usize = 768;
const SCALES: &[usize] = &[1000, 5000, 10000];
const N_QUERIES: usize = 100;
const K: usize = 10;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("  EmergentDB Index Benchmark (Direct Rust, No HTTP)");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("  Dimensions: {}", VECTOR_DIM);
    println!("  Queries: {}", N_QUERIES);
    println!("  Top-K: {}", K);
    println!();

    // Show pre-computed elites grid info
    let grid = PrecomputedElitesGrid::new();
    println!("  Pre-computed Elites Grid: {} industry-standard configurations", grid.len());
    println!("  Sources: OpenSearch, Milvus, Pinecone, ANN-benchmarks");
    println!();

    // Run comprehensive benchmark at 10K scale (tests all grid configurations)
    run_full_benchmark(10_000);

    // Quick comparison at smaller scale
    println!("\n\n═══════════════════════════════════════════════════════════════════════════════════════");
    println!("  QUICK COMPARISON: 5000 vectors");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    run_scale_benchmark(5_000);
}

fn run_scale_benchmark(n: usize) {
    println!("\n══════════════════════════════════════════════════════════════════════");
    println!("  SCALE: {} vectors", n);
    println!("══════════════════════════════════════════════════════════════════════");

    // Generate random vectors
    print!("  Generating {} random vectors... ", n);
    let start = Instant::now();
    let vectors = generate_vectors(n);
    println!("done in {:?}", start.elapsed());

    // Generate queries and compute ground truth
    print!("  Computing ground truth for {} queries... ", N_QUERIES);
    let start = Instant::now();
    let query_indices: Vec<usize> = (0..N_QUERIES.min(n)).collect();
    let ground_truths = compute_ground_truth(&vectors, &query_indices);
    println!("done in {:?}", start.elapsed());

    // Benchmark each index type
    println!("\n  ┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("  │ Index Type      │ Insert (v/s) │ Build (ms)   │ Search (μs)  │ Recall   │");
    println!("  ├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤");

    // 1. Flat Index (baseline)
    benchmark_flat(&vectors, &query_indices, &ground_truths);

    // 2. HNSW Index
    benchmark_hnsw(&vectors, &query_indices, &ground_truths);

    // 3. IVF Index
    if n >= 256 {
        benchmark_ivf(&vectors, &query_indices, &ground_truths, n);
    }

    // 4. EmergentIndex with evolution
    if n >= 100 {
        benchmark_emergent(&vectors, &query_indices, &ground_truths);
    }

    println!("  └─────────────────┴──────────────┴──────────────┴──────────────┴──────────┘");

    // 5. Test preset configurations (skip evolution, use battle-tested configs)
    benchmark_presets(&vectors, &query_indices, &ground_truths, n);
}

fn generate_vectors(n: usize) -> Vec<(NodeId, Embedding)> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let data: Vec<f32> = (0..VECTOR_DIM).map(|_| rng.r#gen::<f32>()).collect();
            let mut emb = Embedding::new(data);
            emb.normalize();
            (NodeId::new(i as u64), emb)
        })
        .collect()
}

fn compute_ground_truth(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
) -> Vec<Vec<NodeId>> {
    // Brute-force ground truth computation
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
) {
    let config = IndexConfig::flat(VECTOR_DIM, DistanceMetric::Cosine);
    let mut index = FlatIndex::new(config);

    // Insert
    let insert_start = Instant::now();
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }
    let insert_time = insert_start.elapsed();
    let insert_throughput = vectors.len() as f64 / insert_time.as_secs_f64();

    // Search
    let (search_times, recalls) = run_searches(&index, vectors, query_indices, ground_truths);
    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    print_row("Flat", insert_throughput, 0.0, avg_search, avg_recall);
}

fn benchmark_hnsw(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
) {
    let config = IndexConfig {
        dim: VECTOR_DIM,
        metric: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    };
    let mut index = HnswIndex::new(config);

    // Insert (also builds the graph)
    let insert_start = Instant::now();
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }
    let insert_time = insert_start.elapsed();
    let insert_throughput = vectors.len() as f64 / insert_time.as_secs_f64();

    // Search
    let (search_times, recalls) = run_searches(&index, vectors, query_indices, ground_truths);
    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    print_row("HNSW (m=16)", insert_throughput, 0.0, avg_search, avg_recall);
}

fn benchmark_ivf(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
    n: usize,
) {
    let n_partitions = (n / 10).max(16).min(256);
    let config = IvfConfig {
        dim: VECTOR_DIM,
        metric: DistanceMetric::Cosine,
        num_partitions: n_partitions,
        nprobe: 16,
        kmeans_iterations: 10,
    };
    let mut index = IvfIndex::new(config);

    // Insert
    let insert_start = Instant::now();
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }
    let insert_time = insert_start.elapsed();

    // Train (compute centroids)
    let build_start = Instant::now();
    index.train().unwrap();
    let build_time = build_start.elapsed();

    let insert_throughput = vectors.len() as f64 / insert_time.as_secs_f64();

    // Search
    let (search_times, recalls) = run_searches(&index, vectors, query_indices, ground_truths);
    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    print_row(&format!("IVF ({})", n_partitions), insert_throughput, build_time.as_millis() as f64, avg_search, avg_recall);
}

fn benchmark_emergent(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
) {
    let config = EmergentConfig {
        dim: VECTOR_DIM,
        metric: DistanceMetric::Cosine,
        ..EmergentConfig::fast()
    };
    let mut index = EmergentIndex::new(config);

    // Insert
    let insert_start = Instant::now();
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }
    let insert_time = insert_start.elapsed();
    let insert_throughput = vectors.len() as f64 / insert_time.as_secs_f64();

    // Evolve (MAP-Elites)
    print!("  │ Evolving...     │              │              │              │          │\r");
    use std::io::Write;
    std::io::stdout().flush().unwrap();

    let evolve_start = Instant::now();
    let elite = index.evolve().unwrap();
    let evolve_time = evolve_start.elapsed();

    // Search with evolved index
    let (search_times, recalls) = run_searches(&index, vectors, query_indices, ground_truths);
    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    // Print result
    let name = format!("Emergent→{:?}", elite.genome.index_type);
    print_row(&name, insert_throughput, evolve_time.as_millis() as f64, avg_search, avg_recall);

    // Print evolution details
    println!("  │   └─ Evolution: fitness={:.3}, internal_recall={:.1}%, internal_latency={:.0}μs",
        elite.fitness,
        elite.metrics.recall_at_10 * 100.0,
        elite.metrics.query_latency_us,
    );
}

fn run_searches<I: VectorIndex>(
    index: &I,
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
) -> (Vec<f64>, Vec<f32>) {
    let mut search_times = Vec::with_capacity(query_indices.len());
    let mut recalls = Vec::with_capacity(query_indices.len());

    for (i, &qi) in query_indices.iter().enumerate() {
        let query = &vectors[qi].1;

        let start = Instant::now();
        let results = index.search(query, K).unwrap();
        let elapsed = start.elapsed().as_micros() as f64;

        search_times.push(elapsed);
        recalls.push(compute_recall(&ground_truths[i], &results));
    }

    (search_times, recalls)
}

fn print_row(name: &str, insert_throughput: f64, build_ms: f64, search_us: f64, recall: f32) {
    println!(
        "  │ {:<15} │ {:>10.0}   │ {:>10.1}   │ {:>10.1}   │ {:>6.1}%  │",
        name,
        insert_throughput,
        build_ms,
        search_us,
        recall * 100.0
    );
}

// ============================================================================
// PRESET BENCHMARK - Test pre-evolved configurations
// ============================================================================

fn benchmark_presets(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
    n: usize,
) {
    println!("\n  ╔═══════════════════════════════════════════════════════════════════════╗");
    println!("  ║  PRESET CONFIGURATIONS (Skip Evolution, Battle-Tested Configs)        ║");
    println!("  ╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Show recommended preset for this scale
    let recommended = IndexPreset::recommend(n, "balanced");
    println!("  Recommended preset for {} vectors: {}", n, recommended);
    println!();

    println!("  ┌───────────────────────────┬──────────────┬──────────────┬──────────┐");
    println!("  │ Preset                    │ Build (ms)   │ Search (μs)  │ Recall   │");
    println!("  ├───────────────────────────┼──────────────┼──────────────┼──────────┤");

    // Test key presets based on scale
    let presets_to_test = if n <= 5000 {
        vec![IndexPreset::Flat, IndexPreset::HnswFast, IndexPreset::HnswBalanced]
    } else {
        vec![IndexPreset::HnswFast, IndexPreset::HnswBalanced, IndexPreset::HnswAccurate]
    };

    for preset in presets_to_test {
        benchmark_single_preset(preset, vectors, query_indices, ground_truths);
    }

    println!("  └───────────────────────────┴──────────────┴──────────────┴──────────┘");
}

fn benchmark_single_preset(
    preset: IndexPreset,
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
) {
    let config = EmergentConfig {
        dim: VECTOR_DIM,
        metric: DistanceMetric::Cosine,
        ..EmergentConfig::fast()
    };
    let mut index = EmergentIndex::new(config);

    // Insert all vectors first
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }

    // Apply preset (this builds the index)
    let build_start = Instant::now();
    if let Err(e) = index.apply_preset(preset) {
        println!("  │ {:<25} │ FAILED: {} │", preset, e);
        return;
    }
    let build_time = build_start.elapsed();

    // Search benchmark
    let (search_times, recalls) = run_searches(&index, vectors, query_indices, ground_truths);
    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    println!(
        "  │ {:<25} │ {:>10.1}   │ {:>10.1}   │ {:>6.1}%  │",
        format!("{}", preset),
        build_time.as_millis() as f64,
        avg_search,
        avg_recall * 100.0
    );
}

// ============================================================================
// PRE-COMPUTED ELITES GRID BENCHMARK
// Tests industry-standard configurations from OpenSearch, Milvus, ANN-benchmarks
// ============================================================================

fn benchmark_precomputed_grid(
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
    n: usize,
) {
    let grid = PrecomputedElitesGrid::new();
    let matching_elites = grid.get_for_scale(n);

    println!("\n  ╔═══════════════════════════════════════════════════════════════════════════════════╗");
    println!("  ║  PRE-COMPUTED ELITES GRID ({} configs for {} vectors)                           ║", matching_elites.len(), n);
    println!("  ║  Source: OpenSearch, Milvus, Pinecone, ANN-benchmarks                             ║");
    println!("  ╚═══════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Show recommended by priority
    println!("  Recommendations by priority:");
    for priority in &["speed", "balanced", "accuracy", "max"] {
        let elite = grid.recommend(n, priority);
        println!("    {:>10}: {} (expected recall: {:.0}%)",
            priority, elite.description, elite.expected_recall * 100.0);
    }
    println!();

    println!("  ┌──────────────────────────────────────────┬──────────────┬──────────────┬──────────┬──────────┐");
    println!("  │ Configuration                            │ Build (ms)   │ Search (μs)  │ Recall   │ Expected │");
    println!("  ├──────────────────────────────────────────┼──────────────┼──────────────┼──────────┼──────────┤");

    // Test each matching configuration
    for elite in matching_elites.iter().take(6) {  // Limit to 6 to keep runtime reasonable
        benchmark_single_elite(elite, vectors, query_indices, ground_truths);
    }

    println!("  └──────────────────────────────────────────┴──────────────┴──────────────┴──────────┴──────────┘");
}

fn benchmark_single_elite(
    elite: &PrecomputedElite,
    vectors: &[(NodeId, Embedding)],
    query_indices: &[usize],
    ground_truths: &[Vec<NodeId>],
) {
    let config = EmergentConfig {
        dim: VECTOR_DIM,
        metric: DistanceMetric::Cosine,
        ..EmergentConfig::fast()
    };
    let mut index = EmergentIndex::new(config);

    // Insert all vectors first
    for (id, emb) in vectors {
        index.insert(*id, emb.clone()).unwrap();
    }

    // Apply pre-computed elite configuration
    let build_start = Instant::now();
    if let Err(e) = index.apply_precomputed_elite(elite) {
        println!("  │ {:<40} │ FAILED: {} │", elite.description, e);
        return;
    }
    let build_time = build_start.elapsed();

    // Search benchmark
    let (search_times, recalls) = run_searches(&index, vectors, query_indices, ground_truths);
    let avg_search = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;

    // Truncate description to fit
    let desc = if elite.description.len() > 38 {
        format!("{}...", &elite.description[..35])
    } else {
        elite.description.to_string()
    };

    println!(
        "  │ {:<40} │ {:>10.1}   │ {:>10.1}   │ {:>6.1}%  │ {:>6.1}%  │",
        desc,
        build_time.as_millis() as f64,
        avg_search,
        avg_recall * 100.0,
        elite.expected_recall * 100.0
    );
}

// ============================================================================
// MAIN: Run all benchmarks
// ============================================================================

fn run_full_benchmark(n: usize) {
    println!("\n══════════════════════════════════════════════════════════════════════════════════════");
    println!("  COMPREHENSIVE BENCHMARK: {} vectors x {} dimensions", n, VECTOR_DIM);
    println!("══════════════════════════════════════════════════════════════════════════════════════");

    // Generate random vectors
    print!("  Generating {} random vectors... ", n);
    let start = Instant::now();
    let vectors = generate_vectors(n);
    println!("done in {:?}", start.elapsed());

    // Generate queries and compute ground truth
    print!("  Computing ground truth for {} queries... ", N_QUERIES);
    let start = Instant::now();
    let query_indices: Vec<usize> = (0..N_QUERIES.min(n)).collect();
    let ground_truths = compute_ground_truth(&vectors, &query_indices);
    println!("done in {:?}", start.elapsed());

    // Run standard index benchmarks
    println!("\n  ┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("  │ Index Type      │ Insert (v/s) │ Build (ms)   │ Search (μs)  │ Recall   │");
    println!("  ├─────────────────┼──────────────┼──────────────┼──────────────┼──────────┤");

    benchmark_flat(&vectors, &query_indices, &ground_truths);

    if n >= 256 {
        benchmark_ivf(&vectors, &query_indices, &ground_truths, n);
    }

    println!("  └─────────────────┴──────────────┴──────────────┴──────────────┴──────────┘");

    // Run pre-computed elites grid benchmark
    benchmark_precomputed_grid(&vectors, &query_indices, &ground_truths, n);
}
