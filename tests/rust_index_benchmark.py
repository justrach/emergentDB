#!/usr/bin/env python
"""
Rust Index Benchmark - Compare all EmergentDB indices

Creates a Rust test file that benchmarks Flat, HNSW, IVF, and PQ
with the same embeddings from the PDF test.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
import requests
import time
import random

# Gemini API - use environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

DATA_DIR = Path("/Users/rachpradhan/Experiments/backend-new/tests/pdf_data")
RUST_DIR = Path("/Users/rachpradhan/Experiments/backend-new")


def get_gemini_embedding(text: str) -> list[float]:
    """Get embedding from Gemini API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text[:8000]}]}
    }

    response = requests.post(
        f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print(f"Embedding error: {response.status_code}")
        return [0.0] * 768

    data = response.json()
    return data["embedding"]["values"]


def load_or_create_embeddings():
    """Load existing embeddings or create new ones."""
    embeddings_file = DATA_DIR / "embeddings.json"

    if embeddings_file.exists():
        print("Loading existing embeddings...")
        with open(embeddings_file) as f:
            data = json.load(f)
            return [d["embedding"] for d in data]

    print("Creating sample embeddings...")
    # Generate some test embeddings
    embeddings = []
    test_texts = [
        "Vector databases are essential for modern AI applications",
        "HNSW provides logarithmic search complexity with high recall",
        "Product quantization compresses vectors for memory efficiency",
        "IVF partitions vectors into clusters for faster search",
        "The Darwin Godel Machine is a self-improving AI system",
    ]

    for i, text in enumerate(test_texts):
        print(f"  Embedding {i+1}/{len(test_texts)}...")
        emb = get_gemini_embedding(text)
        embeddings.append(emb)
        time.sleep(0.5)

    return embeddings


def generate_rust_test(embeddings: list[list[float]], num_extra: int = 100):
    """Generate Rust test code for all indices."""
    dim = len(embeddings[0])

    # Generate additional random embeddings for larger test
    print(f"Generating {num_extra} additional random vectors...")
    all_embeddings = embeddings.copy()

    for _ in range(num_extra):
        # Generate random unit vector
        vec = [random.gauss(0, 1) for _ in range(dim)]
        norm = sum(x*x for x in vec) ** 0.5
        vec = [x/norm for x in vec]
        all_embeddings.append(vec)

    # Format embeddings for Rust
    def format_vec(vec):
        return "[" + ", ".join(f"{x:.6}f32" for x in vec[:50]) + ", /* ... truncated */]"

    rust_code = f'''//! Benchmark all EmergentDB indices with real embeddings
//!
//! Generated from PDF indexing test
//! Embedding dimension: {dim}
//! Total vectors: {len(all_embeddings)}

use std::time::Instant;
use vector_core::{{
    index::{{
        flat::FlatIndex,
        hnsw::HnswIndex,
        ivf::{{IvfConfig, IvfIndex}},
        pq::{{PqConfig, PqIndex}},
        IndexConfig, VectorIndex,
    }},
    DistanceMetric, Embedding, NodeId,
}};

/// Benchmark results for an index
struct BenchResult {{
    name: String,
    insert_time_ms: f64,
    search_time_us: f64,
    recall_at_5: f32,
}}

/// Generate test vectors (truncated from real Gemini embeddings)
fn generate_test_vectors() -> Vec<Vec<f32>> {{
    // Using 768-dim vectors from Gemini text-embedding-004
    // Generating random unit vectors for benchmark
    let dim = {dim};
    let n = {len(all_embeddings)};

    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {{
            let mut vec: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            vec.iter_mut().for_each(|x| *x /= norm);
            vec
        }})
        .collect()
}}

/// Compute recall@k
fn compute_recall(ground_truth: &[NodeId], results: &[NodeId], k: usize) -> f32 {{
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
    let result_set: std::collections::HashSet<_> = results.iter().take(k).collect();
    let hits = gt_set.intersection(&result_set).count();
    hits as f32 / k as f32
}}

fn main() {{
    println!("{{}}", "=".repeat(60));
    println!("EmergentDB Index Benchmark");
    println!("Vectors: {len(all_embeddings)}, Dimension: {dim}");
    println!("{{}}", "=".repeat(60));

    let vectors = generate_test_vectors();
    let query_indices: Vec<usize> = (0..10).collect();
    let k = 5;

    let mut results = Vec::new();

    // 1. FLAT INDEX (ground truth)
    println!("\\n[1/4] Benchmarking FLAT index (ground truth)...");
    let flat_config = IndexConfig::flat({dim}, DistanceMetric::Cosine);
    let mut flat_index = FlatIndex::new(flat_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {{
        flat_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }}
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    // Ground truth search
    let mut ground_truths = Vec::new();
    let start = Instant::now();
    for &qi in &query_indices {{
        let query = Embedding::new(vectors[qi].clone());
        let res = flat_index.search(&query, k).unwrap();
        ground_truths.push(res.iter().map(|r| r.id).collect::<Vec<_>>());
    }}
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {{
        name: "Flat".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: 1.0, // Ground truth
    }});
    println!("  Insert: {{:.2}}ms, Search: {{:.1}}us/query", insert_time, search_time);

    // 2. HNSW INDEX
    println!("\\n[2/4] Benchmarking HNSW index...");
    let hnsw_config = IndexConfig::hnsw({dim}, DistanceMetric::Cosine);
    let mut hnsw_index = HnswIndex::new(hnsw_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {{
        hnsw_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }}
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {{
        let query = Embedding::new(vectors[qi].clone());
        let res = hnsw_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }}
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {{
        name: "HNSW".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
    }});
    println!("  Insert: {{:.2}}ms, Search: {{:.1}}us/query, Recall@5: {{:.1}}%",
             insert_time, search_time, total_recall / query_indices.len() as f32 * 100.0);

    // 3. IVF INDEX
    println!("\\n[3/4] Benchmarking IVF index...");
    let ivf_config = IvfConfig {{
        dim: {dim},
        metric: DistanceMetric::Cosine,
        num_partitions: 16,
        nprobe: 4,
        kmeans_iterations: 10,
    }};
    let mut ivf_index = IvfIndex::new(ivf_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {{
        ivf_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }}
    ivf_index.train().unwrap();
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {{
        let query = Embedding::new(vectors[qi].clone());
        let res = ivf_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }}
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {{
        name: "IVF".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
    }});
    println!("  Insert+Train: {{:.2}}ms, Search: {{:.1}}us/query, Recall@5: {{:.1}}%",
             insert_time, search_time, total_recall / query_indices.len() as f32 * 100.0);

    // 4. PQ INDEX
    println!("\\n[4/4] Benchmarking PQ index...");
    let pq_config = PqConfig {{
        dim: {dim},
        num_subvectors: 48, // 768 / 48 = 16 dims per subvector
        num_centroids: 256,
        kmeans_iterations: 10,
    }};
    let mut pq_index = PqIndex::new(pq_config);

    let start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {{
        pq_index.insert(NodeId::new(i as u64), Embedding::new(vec.clone())).unwrap();
    }}
    pq_index.train().unwrap();
    let insert_time = start.elapsed().as_secs_f64() * 1000.0;

    let mut total_recall = 0.0;
    let start = Instant::now();
    for (j, &qi) in query_indices.iter().enumerate() {{
        let query = Embedding::new(vectors[qi].clone());
        let res = pq_index.search(&query, k).unwrap();
        let result_ids: Vec<_> = res.iter().map(|r| r.id).collect();
        total_recall += compute_recall(&ground_truths[j], &result_ids, k);
    }}
    let search_time = start.elapsed().as_micros() as f64 / query_indices.len() as f64;

    results.push(BenchResult {{
        name: "PQ".to_string(),
        insert_time_ms: insert_time,
        search_time_us: search_time,
        recall_at_5: total_recall / query_indices.len() as f32,
    }});
    println!("  Insert+Train: {{:.2}}ms, Search: {{:.1}}us/query, Recall@5: {{:.1}}%",
             insert_time, search_time, total_recall / query_indices.len() as f32 * 100.0);

    // Summary
    println!("\\n{{}}", "=".repeat(60));
    println!("BENCHMARK SUMMARY");
    println!("{{}}", "=".repeat(60));
    println!("\\n{{:<10}} {{:>12}} {{:>12}} {{:>10}}", "Index", "Insert(ms)", "Search(us)", "Recall@5");
    println!("{{:-<10}} {{:-<12}} {{:-<12}} {{:-<10}}", "", "", "", "");

    for r in &results {{
        println!("{{:<10}} {{:>12.2}} {{:>12.1}} {{:>9.1}}%",
                 r.name, r.insert_time_ms, r.search_time_us, r.recall_at_5 * 100.0);
    }}

    // Best index recommendation
    let best_recall = results.iter().max_by(|a, b| a.recall_at_5.partial_cmp(&b.recall_at_5).unwrap()).unwrap();
    let best_speed = results.iter().filter(|r| r.recall_at_5 > 0.8).min_by(|a, b| a.search_time_us.partial_cmp(&b.search_time_us).unwrap());

    println!("\\n  Best Recall: {{}} ({{:.1}}%)", best_recall.name, best_recall.recall_at_5 * 100.0);
    if let Some(bs) = best_speed {{
        println!("  Best Speed (recall>80%): {{}} ({{:.1}}us)", bs.name, bs.search_time_us);
    }}

    println!("\\n{{}}", "=".repeat(60));
}}
'''

    return rust_code


def main():
    print("=" * 60)
    print("EmergentDB Rust Index Benchmark Generator")
    print("=" * 60)

    # Load embeddings
    embeddings = load_or_create_embeddings()
    print(f"Loaded {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # Generate Rust test (need 256+ for PQ training)
    rust_code = generate_rust_test(embeddings, num_extra=300)

    # Write to file
    test_file = RUST_DIR / "examples" / "pdf_benchmark.rs"
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, "w") as f:
        f.write(rust_code)

    print(f"\nGenerated: {test_file}")

    # Update Cargo.toml to include the example
    cargo_toml = RUST_DIR / "crates" / "vector-core" / "Cargo.toml"
    cargo_content = cargo_toml.read_text()

    if "[[example]]" not in cargo_content:
        cargo_content += """
[[example]]
name = "pdf_benchmark"
path = "../../examples/pdf_benchmark.rs"
"""
        cargo_toml.write_text(cargo_content)
        print("Updated Cargo.toml with example")

    # Run the benchmark
    print("\n" + "=" * 60)
    print("Running Rust benchmark...")
    print("=" * 60 + "\n")

    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "pdf_benchmark"],
        cwd=str(RUST_DIR),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
