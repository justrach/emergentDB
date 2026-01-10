//! Benchmark all EmergentDB indices with real embeddings
//!
//! Compares:
//! - Manual index selection (Flat, HNSW, IVF, PQ)
//! - Emergent auto-selection via MAP-Elites evolution
//!
//! Embedding dimension: 768
//! Total vectors: 1000 (for Emergent evolution)

use std::time::Instant;
use vector_core::{
    index::{
        flat::FlatIndex,
        hnsw::HnswIndex,
        ivf::{IvfConfig, IvfIndex},
        pq::{PqConfig, PqIndex},
        emergent::{EmergentConfig, EmergentIndex},
        IndexConfig, VectorIndex,
    },
    DistanceMetric, Embedding, NodeId,
};

/// Benchmark results for an index
struct BenchResult {
    name: String,
    insert_time_ms: f64,
    search_time_us: f64,
    recall_at_5: f32,
    auto_selected: Option<String>,
}

/// Generate test vectors (random unit vectors)
fn generate_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
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

/// Compute recall@k
fn compute_recall(ground_truth: &[NodeId], results: &[NodeId], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
    let result_set: std::collections::HashSet<_> = results.iter().take(k).collect();
    let hits = gt_set.intersection(&result_set).count();
    hits as f32 / k as f32
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("EmergentDB Index Benchmark: Manual vs Auto Selection");
    println!("{}", "=".repeat(70));

    let dim = 768;
    let n_vectors = 1000; // Enough for Emergent evolution
    let vectors = generate_test_vectors(n_vectors, dim);
    let query_indices: Vec<usize> = (0..20).collect();
    let k = 5;

    println!("\nDataset: {} vectors, {} dimensions", n_vectors, dim);
    println!("Queries: {}, k={}", query_indices.len(), k);

    let mut results = Vec::new();

    // =========================================================================
    // MANUAL INDEX SELECTION
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("PART 1: MANUAL INDEX SELECTION");
    println!("{}", "=".repeat(70));

    // 1. FLAT INDEX (ground truth)
    println!("\n[1/5] Benchmarking FLAT index (ground truth)...");
    let flat_config = IndexConfig::flat(dim, DistanceMetric::Cosine);
    let mut flat_index = FlatIndex::new(flat_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        flat_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    // Ground truth search
    let mut ground_truths = Vec::new();
    let start = Instant::now();
    for &qi in &query_indices {
        let query = Embedding::new(vectors[qi].clone());
        let res = flat_index.search(&query, k).unwrap();
        ground_truths.push(res.iter().map(|r| r.id).collect::<Vec<_>>());
    }
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {
        name: "Flat (Manual)".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: 1.0,
        auto_selected: None,
    });
    println!("  Insert: {:.2}ms, Search: {:.1}us/query", insert_time, search_time);

    // 2. HNSW INDEX
    println!("\n[2/5] Benchmarking HNSW index...");
    let hnsw_config = IndexConfig::hnsw(dim, DistanceMetric::Cosine);
    let mut hnsw_index = HnswIndex::new(hnsw_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        hnsw_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {
        let query = Embedding::new(vectors[qi].clone());
        let res = hnsw_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {
        name: "HNSW (Manual)".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
        auto_selected: None,
    });
    println!("  Insert: {:.2}ms, Search: {:.1}us/query, Recall@5: {:.1}%",
             insert_time, search_time, total_recall / query_indices.len() as f32 * 100.0);

    // 3. IVF INDEX
    println!("\n[3/5] Benchmarking IVF index...");
    let ivf_config = IvfConfig {
        dim,
        metric: DistanceMetric::Cosine,
        num_partitions: 32,
        nprobe: 8,
        kmeans_iterations: 10,
    };
    let mut ivf_index = IvfIndex::new(ivf_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        ivf_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    ivf_index.train().unwrap();
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {
        let query = Embedding::new(vectors[qi].clone());
        let res = ivf_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {
        name: "IVF (Manual)".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
        auto_selected: None,
    });
    println!("  Insert+Train: {:.2}ms, Search: {:.1}us/query, Recall@5: {:.1}%",
             insert_time, search_time, total_recall / query_indices.len() as f32 * 100.0);

    // 4. PQ INDEX
    println!("\n[4/5] Benchmarking PQ index...");
    let pq_config = PqConfig {
        dim,
        num_subvectors: 48,
        num_centroids: 256,
        kmeans_iterations: 10,
    };
    let mut pq_index = PqIndex::new(pq_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        pq_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    pq_index.train().unwrap();
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {
        let query = Embedding::new(vectors[qi].clone());
        let res = pq_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {
        name: "PQ (Manual)".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
        auto_selected: None,
    });
    println!("  Insert+Train: {:.2}ms, Search: {:.1}us/query, Recall@5: {:.1}%",
             insert_time, search_time, total_recall / query_indices.len() as f32 * 100.0);

    // =========================================================================
    // EMERGENT AUTO-SELECTION (THE WHOLE POINT!)
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("PART 2: EMERGENT AUTO-SELECTION (MAP-Elites Evolution)");
    println!("{}", "=".repeat(70));

    // 5. EMERGENT INDEX - Auto-selects best index via evolution
    println!("\n[5/5] Benchmarking EMERGENT index (auto-selection)...");
    println!("  Running MAP-Elites evolution with 3D behavior grid (recall x latency x memory)...");
    println!("  Using search_first priority (speed=55%, recall=35%, build=5%)...");

    // Use search_first() preset for best query latency
    let mut emergent_config = EmergentConfig::search_first();
    emergent_config.dim = dim;
    emergent_config.metric = DistanceMetric::Cosine;

    let mut emergent_index = EmergentIndex::new(emergent_config);

    // Insert all vectors
    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        emergent_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }
    let insert_time_pre_evolve = start.elapsed().as_secs_f64() * 1000.0;

    // Run evolution to auto-select best index
    let evolve_start = Instant::now();
    let elite = emergent_index.evolve().unwrap();
    let evolve_time = evolve_start.elapsed().as_secs_f64() * 1000.0;

    let total_insert_time = insert_time_pre_evolve + evolve_time;
    let selected_type = format!("{}", elite.genome.index_type);

    println!("  Evolution completed in {:.1}ms", evolve_time);
    println!("  Selected index type: {} (evolved from MAP-Elites)", selected_type);
    println!("  Archive coverage: {:.1}% (3D grid: recall x latency x memory)", emergent_index.archive_coverage());
    println!("  Elite fitness: {:.3} (geometric mean of 4 metrics)", elite.fitness);
    println!("  Elite metrics:");
    println!("    - Recall@10: {:.1}%", elite.metrics.recall_at_10 * 100.0);
    println!("    - Latency: {:.1}us", elite.metrics.query_latency_us);
    println!("    - Throughput: {:.0} QPS", elite.metrics.throughput_qps);
    println!("    - Build time: {:.1}ms", elite.metrics.build_time_ms);
    println!("    - Memory: {:.0} bytes/vec", elite.metrics.bytes_per_vector);

    // InsertQD results (Dual-QD system)
    println!("\n  InsertQD (Insert Strategy Evolution):");
    if let Some(insert_elite) = emergent_index.get_best_insert_elite() {
        println!("    Selected strategy: {}", insert_elite.genome.strategy);
        println!("    Throughput: {:.0} vectors/sec", insert_elite.metrics.throughput_vps);
        println!("    CPU efficiency: {:.1}%", insert_elite.metrics.cpu_efficiency * 100.0);
        println!("    Insert archive coverage: {:.1}%", emergent_index.insert_archive_coverage());
        println!("    Insert fitness: {:.3}", insert_elite.fitness);
    } else {
        println!("    (InsertQD not run - requires 100+ vectors)");
    }

    // Benchmark search with evolved index
    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {
        let query = Embedding::new(vectors[qi].clone());
        let res = emergent_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {
        name: "Emergent (Auto)".to_string(),
        insert_time_ms: total_insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
        auto_selected: Some(selected_type.clone()),
    });
    println!("  Total time: {:.2}ms (insert + evolve)", total_insert_time);
    println!("  Search: {:.1}us/query, Recall@5: {:.1}%",
             search_time, total_recall / query_indices.len() as f32 * 100.0);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK SUMMARY: Manual vs Auto Selection");
    println!("{}", "=".repeat(70));
    println!("\n{:<20} {:>12} {:>12} {:>10} {:>15}",
             "Index", "Insert(ms)", "Search(us)", "Recall@5", "Auto-Selected");
    println!("{}", "-".repeat(70));

    for r in &results {
        let auto = r.auto_selected.as_deref().unwrap_or("-");
        println!("{:<20} {:>12.2} {:>12.1} {:>9.1}% {:>15}",
                 r.name, r.insert_time_ms, r.search_time_us, r.recall_at_5 * 100.0, auto);
    }

    // Find best in each category
    println!("\n{}", "-".repeat(70));

    let manual_results: Vec<_> = results.iter().filter(|r| r.auto_selected.is_none()).collect();
    let auto_result = results.iter().find(|r| r.auto_selected.is_some()).unwrap();

    let best_manual_recall = manual_results.iter()
        .max_by(|a, b| a.recall_at_5.partial_cmp(&b.recall_at_5).unwrap()).unwrap();
    let best_manual_speed = manual_results.iter()
        .filter(|r| r.recall_at_5 > 0.8)
        .min_by(|a, b| a.search_time_us.partial_cmp(&b.search_time_us).unwrap());

    println!("\n  MANUAL SELECTION:");
    println!("    Best Recall: {} ({:.1}%)", best_manual_recall.name, best_manual_recall.recall_at_5 * 100.0);
    if let Some(bs) = best_manual_speed {
        println!("    Best Speed (recall>80%): {} ({:.1}us)", bs.name, bs.search_time_us);
    }

    println!("\n  AUTO SELECTION (Emergent):");
    println!("    Selected: {} via MAP-Elites evolution", auto_result.auto_selected.as_ref().unwrap());
    println!("    Recall: {:.1}%", auto_result.recall_at_5 * 100.0);
    println!("    Search latency: {:.1}us", auto_result.search_time_us);

    // Compare auto vs best manual
    if auto_result.recall_at_5 >= best_manual_recall.recall_at_5 * 0.95 {
        println!("\n  ✓ Emergent matched/exceeded best manual recall!");
    }
    if let Some(bs) = best_manual_speed {
        if auto_result.search_time_us <= bs.search_time_us * 1.2 {
            println!("  ✓ Emergent matched best manual speed (within 20%)!");
        }
    }

    println!("\n{}", "=".repeat(70));
}
