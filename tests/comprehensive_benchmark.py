#!/usr/bin/env python3
"""
EmergentDB Comprehensive Benchmark
===================================

Compares EmergentDB (all index modes) vs LanceDB vs ChromaDB

Tests:
- Insertion throughput (vectors/sec)
- Search latency (μs)
- Recall@10 accuracy
- Memory efficiency

Index modes tested:
- EmergentDB: Flat, HNSW, IVF, Emergent (auto-optimized)
- LanceDB: Flat, IVF_PQ, IVF_HNSW_SQ
- ChromaDB: HNSW (default)

Scales: 1K, 10K, 50K vectors
Dimensions: 768 (Gemini embedding size)
"""

import json
import time
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# Check dependencies
try:
    import lancedb
    import chromadb
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install lancedb chromadb requests numpy")
    sys.exit(1)

# Configuration
EMERGENTDB_URL = "http://localhost:3000"
OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
VECTOR_DIM = 768
SCALES = [1000, 5000, 10000]  # Reduced scales for faster benchmarking
N_QUERIES = 50
K = 10  # top-k for recall


@dataclass
class BenchmarkResult:
    database: str
    index_type: str
    n_vectors: int
    insert_time_ms: float
    insert_throughput: float  # vectors/sec
    search_time_us: float
    recall_at_10: float
    notes: str = ""


def generate_vectors(n: int, dim: int = VECTOR_DIM) -> tuple[np.ndarray, list[str]]:
    """Generate random normalized vectors."""
    print(f"    Generating {n:,} random vectors...")
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    ids = [str(i) for i in range(n)]
    return vectors, ids


def compute_ground_truth(vectors: np.ndarray, query_indices: list[int], k: int = K) -> list[list[str]]:
    """Compute brute-force ground truth using cosine similarity."""
    ground_truths = []
    for qi in query_indices:
        query = vectors[qi]
        similarities = np.dot(vectors, query)
        top_k = np.argsort(similarities)[::-1][:k]
        ground_truths.append([str(i) for i in top_k])
    return ground_truths


def compute_recall(ground_truth: list[str], results: list[str], k: int = K) -> float:
    """Compute recall@k."""
    gt_set = set(ground_truth[:k])
    result_set = set(results[:k])
    return len(gt_set & result_set) / k


# =============================================================================
# EMERGENTDB BENCHMARKS (via REST API)
# =============================================================================

def check_emergentdb_running() -> bool:
    """Check if EmergentDB server is running."""
    try:
        r = requests.get(f"{EMERGENTDB_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def configure_emergentdb_index(index_type: str, params: dict = None) -> bool:
    """Configure EmergentDB to use a specific index type."""
    payload = {"index_type": index_type}
    if params:
        payload.update(params)

    try:
        r = requests.post(
            f"{EMERGENTDB_URL}/index/configure",
            json=payload,
            timeout=120
        )
        if r.status_code == 200:
            result = r.json()
            return result.get("success", False)
    except:
        pass
    return False


def reset_emergentdb_index() -> bool:
    """Reset EmergentDB index to brute-force mode."""
    try:
        r = requests.post(f"{EMERGENTDB_URL}/index/reset", timeout=10)
        return r.status_code == 200
    except:
        return False


def benchmark_emergentdb(
    vectors: np.ndarray,
    ids: list[str],
    query_indices: list[int],
    ground_truths: list[list[str]],
    index_type: str = "flat",
    index_params: dict = None,
    evolve: bool = False,
    skip_insert: bool = False
) -> Optional[BenchmarkResult]:
    """Benchmark EmergentDB via REST API."""
    n = len(vectors)
    mode = "EVOLVED" if evolve else index_type.upper()
    print(f"\n  [EmergentDB {mode}] {n:,} vectors...")

    if not check_emergentdb_running():
        print("    ERROR: EmergentDB server not running. Start with:")
        print("    DATA_DIR=./benchmark_data VECTOR_DIM=768 cargo run --release -p api-server")
        return None

    try:
        # Insert vectors (skip if already inserted for same scale)
        insert_time = 0.0
        if not skip_insert:
            batch_size = 500
            start = time.perf_counter()

            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch = [
                    {"id": int(ids[j]), "vector": vectors[j].tolist()}
                    for j in range(i, end)
                ]
                r = requests.post(
                    f"{EMERGENTDB_URL}/vectors/batch_insert",
                    json={"vectors": batch},
                    timeout=120
                )
                if r.status_code != 200:
                    print(f"    Insert failed: {r.text}")
                    return None

            insert_time = (time.perf_counter() - start) * 1000

        insert_throughput = n / (insert_time / 1000) if insert_time > 0 else 0

        # Configure index type (if not evolving)
        build_time = 0.0
        evolution_info = ""

        if evolve and n >= 100:
            print(f"    Running MAP-Elites evolution...")
            evolve_start = time.perf_counter()
            r = requests.post(
                f"{EMERGENTDB_URL}/qd/evolve",
                json={},
                timeout=300
            )
            build_time = (time.perf_counter() - evolve_start) * 1000

            if r.status_code == 200:
                result = r.json()
                if result.get("success"):
                    evolution_info = f" → {result['best_index_type']}"
                    print(f"    Evolved to {result['best_index_type']} in {build_time:.0f}ms")
                    print(f"      Fitness: {result['fitness']:.3f}, Coverage: {result['archive_coverage']:.1f}%")
                else:
                    print(f"    Evolution failed: {result.get('error', 'unknown')}")
            else:
                print(f"    Evolution request failed: {r.text}")
        elif index_type != "brute":
            # Configure specific index type
            print(f"    Building {index_type.upper()} index...")
            build_start = time.perf_counter()
            payload = {"index_type": index_type}
            if index_params:
                payload.update(index_params)

            r = requests.post(
                f"{EMERGENTDB_URL}/index/configure",
                json=payload,
                timeout=120
            )
            build_time = (time.perf_counter() - build_start) * 1000

            if r.status_code == 200:
                result = r.json()
                if result.get("success"):
                    print(f"    Built {index_type.upper()} in {result['build_time_ms']}ms")
                else:
                    print(f"    Build failed: {result.get('error', 'unknown')}")

        # Search benchmark
        search_times = []
        total_recall = 0.0

        for i, qi in enumerate(query_indices):
            query = vectors[qi].tolist()
            start = time.perf_counter()
            r = requests.post(
                f"{EMERGENTDB_URL}/vectors/search",
                json={"query": query, "k": K},
                timeout=10
            )
            search_times.append((time.perf_counter() - start) * 1_000_000)

            if r.status_code == 200:
                results = r.json()["results"]
                result_ids = [str(res["id"]) for res in results]
                total_recall += compute_recall(ground_truths[i], result_ids, K)

        avg_search = sum(search_times) / len(search_times)
        avg_recall = total_recall / len(query_indices)

        if not skip_insert:
            print(f"    Insert: {insert_time:.1f}ms ({insert_throughput:.0f} vec/s)")
        print(f"    Search: {avg_search:.1f}μs, Recall@{K}: {avg_recall*100:.1f}%")

        # Determine final index type name
        if evolve:
            final_type = f"evolved{evolution_info}"
            notes = "MAP-Elites auto-optimized"
        elif index_type == "brute":
            final_type = "brute"
            notes = "SIMD brute-force (no index)"
        else:
            final_type = index_type
            notes = f"{index_type.upper()} index"

        return BenchmarkResult(
            database="EmergentDB",
            index_type=final_type,
            n_vectors=n,
            insert_time_ms=insert_time + build_time,
            insert_throughput=insert_throughput,
            search_time_us=avg_search,
            recall_at_10=avg_recall,
            notes=notes
        )
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# LANCEDB BENCHMARKS
# =============================================================================

def benchmark_lancedb(
    vectors: np.ndarray,
    ids: list[str],
    query_indices: list[int],
    ground_truths: list[list[str]],
    index_type: str = "flat"
) -> BenchmarkResult:
    """Benchmark LanceDB with different index types."""
    n = len(vectors)
    print(f"\n  [LanceDB {index_type.upper()}] {n:,} vectors...")

    tmp_dir = tempfile.mkdtemp()
    try:
        db = lancedb.connect(tmp_dir)
        data = [{"id": ids[i], "vector": vectors[i].tolist()} for i in range(n)]

        # Insert
        start = time.perf_counter()
        table = db.create_table("vectors", data)

        # Create index based on type
        if index_type == "ivf_pq" and n >= 1000:
            num_partitions = min(256, n // 10)
            num_sub_vectors = min(96, VECTOR_DIM // 8)
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                vector_column_name="vector",
                index_type="IVF_PQ"
            )
        elif index_type == "ivf_hnsw" and n >= 1000:
            num_partitions = min(128, n // 10)
            table.create_index(
                metric="cosine",
                num_partitions=num_partitions,
                vector_column_name="vector",
                index_type="IVF_HNSW_SQ",
                m=16,
                ef_construction=150
            )

        insert_time = (time.perf_counter() - start) * 1000
        insert_throughput = n / (insert_time / 1000)

        # Search
        search_times = []
        total_recall = 0.0

        for i, qi in enumerate(query_indices):
            query = vectors[qi].tolist()
            start = time.perf_counter()
            results = table.search(query).limit(K).to_list()
            search_times.append((time.perf_counter() - start) * 1_000_000)

            result_ids = [r["id"] for r in results]
            total_recall += compute_recall(ground_truths[i], result_ids, K)

        avg_search = sum(search_times) / len(search_times)
        avg_recall = total_recall / len(query_indices)

        print(f"    Insert: {insert_time:.1f}ms ({insert_throughput:.0f} vec/s)")
        print(f"    Search: {avg_search:.1f}μs, Recall@{K}: {avg_recall*100:.1f}%")

        return BenchmarkResult(
            database="LanceDB",
            index_type=index_type,
            n_vectors=n,
            insert_time_ms=insert_time,
            insert_throughput=insert_throughput,
            search_time_us=avg_search,
            recall_at_10=avg_recall,
            notes="Arrow-based columnar storage"
        )
    finally:
        shutil.rmtree(tmp_dir)


# =============================================================================
# CHROMADB BENCHMARKS
# =============================================================================

def benchmark_chromadb(
    vectors: np.ndarray,
    ids: list[str],
    query_indices: list[int],
    ground_truths: list[list[str]],
    hnsw_params: dict = None
) -> BenchmarkResult:
    """Benchmark ChromaDB with HNSW index."""
    n = len(vectors)
    index_desc = "hnsw" if not hnsw_params else f"hnsw_m{hnsw_params.get('M', 16)}"
    print(f"\n  [ChromaDB {index_desc.upper()}] {n:,} vectors...")

    tmp_dir = tempfile.mkdtemp()
    try:
        client = chromadb.PersistentClient(path=tmp_dir)

        # Configure HNSW parameters
        metadata = {"hnsw:space": "cosine"}
        if hnsw_params:
            for k, v in hnsw_params.items():
                metadata[f"hnsw:{k}"] = v

        collection = client.create_collection(
            name="vectors",
            metadata=metadata
        )

        # Insert (batch for speed)
        batch_size = 5000
        start = time.perf_counter()
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            collection.add(
                ids=ids[i:end],
                embeddings=[v.tolist() for v in vectors[i:end]]
            )
        insert_time = (time.perf_counter() - start) * 1000
        insert_throughput = n / (insert_time / 1000)

        # Search
        search_times = []
        total_recall = 0.0

        for i, qi in enumerate(query_indices):
            query = vectors[qi].tolist()
            start = time.perf_counter()
            results = collection.query(query_embeddings=[query], n_results=K)
            search_times.append((time.perf_counter() - start) * 1_000_000)

            result_ids = results["ids"][0]
            total_recall += compute_recall(ground_truths[i], result_ids, K)

        avg_search = sum(search_times) / len(search_times)
        avg_recall = total_recall / len(query_indices)

        print(f"    Insert: {insert_time:.1f}ms ({insert_throughput:.0f} vec/s)")
        print(f"    Search: {avg_search:.1f}μs, Recall@{K}: {avg_recall*100:.1f}%")

        return BenchmarkResult(
            database="ChromaDB",
            index_type=index_desc,
            n_vectors=n,
            insert_time_ms=insert_time,
            insert_throughput=insert_throughput,
            search_time_us=avg_search,
            recall_at_10=avg_recall,
            notes="HNSW-based with SQLite metadata"
        )
    finally:
        shutil.rmtree(tmp_dir)


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_scale_benchmark(n_vectors: int) -> list[BenchmarkResult]:
    """Run benchmarks at a specific scale."""
    print(f"\n{'='*70}")
    print(f"SCALE TEST: {n_vectors:,} vectors, {VECTOR_DIM} dimensions")
    print(f"{'='*70}")

    # Generate data
    vectors, ids = generate_vectors(n_vectors)

    # Query setup
    query_indices = list(range(min(N_QUERIES, n_vectors)))
    print(f"    Computing ground truth for {len(query_indices)} queries...")
    ground_truths = compute_ground_truth(vectors, query_indices)

    results = []

    # EmergentDB (if running) - Test brute force and evolved mode
    if check_emergentdb_running():
        # 1. Brute-force (no index) - baseline showing raw SIMD performance
        result = benchmark_emergentdb(
            vectors, ids, query_indices, ground_truths,
            index_type="brute"
        )
        if result:
            results.append(result)

        # 2. MAP-Elites evolved (auto-optimized) - the core feature
        # Evolution will test HNSW, IVF, Flat internally and pick the best
        if n_vectors >= 100:
            reset_emergentdb_index()
            result = benchmark_emergentdb(
                vectors, ids, query_indices, ground_truths,
                evolve=True,
                skip_insert=True
            )
            if result:
                results.append(result)
    else:
        print("\n  [EmergentDB] Skipped - server not running")

    # LanceDB benchmarks
    results.append(benchmark_lancedb(vectors, ids, query_indices, ground_truths, "flat"))
    if n_vectors >= 1000:
        results.append(benchmark_lancedb(vectors, ids, query_indices, ground_truths, "ivf_pq"))
        results.append(benchmark_lancedb(vectors, ids, query_indices, ground_truths, "ivf_hnsw"))

    # ChromaDB benchmarks
    results.append(benchmark_chromadb(vectors, ids, query_indices, ground_truths))
    # ChromaDB with tuned HNSW
    results.append(benchmark_chromadb(
        vectors, ids, query_indices, ground_truths,
        {"M": 32, "construction_ef": 200, "search_ef": 100}
    ))

    return results


def print_summary(all_results: list[BenchmarkResult]):
    """Print benchmark summary."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    for scale in SCALES:
        scale_results = [r for r in all_results if r.n_vectors == scale]
        if not scale_results:
            continue

        print(f"\n  @ {scale:,} vectors:")
        print(f"  {'Database':<15} {'Index':<12} {'Search(μs)':>12} {'Insert(v/s)':>12} {'Recall@10':>10}")
        print(f"  {'-'*65}")

        # Sort by search time
        scale_results.sort(key=lambda x: x.search_time_us)

        for r in scale_results:
            print(f"  {r.database:<15} {r.index_type:<12} {r.search_time_us:>12.1f} {r.insert_throughput:>12.0f} {r.recall_at_10*100:>9.1f}%")

        # Find winners
        fastest = min(scale_results, key=lambda x: x.search_time_us)
        most_accurate = max(scale_results, key=lambda x: x.recall_at_10)
        fastest_insert = max(scale_results, key=lambda x: x.insert_throughput)

        print(f"\n  Winners:")
        print(f"    Fastest search: {fastest.database} ({fastest.index_type}) - {fastest.search_time_us:.1f}μs")
        print(f"    Best recall:    {most_accurate.database} ({most_accurate.index_type}) - {most_accurate.recall_at_10*100:.1f}%")
        print(f"    Fastest insert: {fastest_insert.database} ({fastest_insert.index_type}) - {fastest_insert.insert_throughput:.0f} vec/s")

        # Speedup calculations
        emergent = next((r for r in scale_results if r.database == "EmergentDB"), None)
        lancedb_flat = next((r for r in scale_results if r.database == "LanceDB" and r.index_type == "flat"), None)
        chromadb_default = next((r for r in scale_results if r.database == "ChromaDB" and r.index_type == "hnsw"), None)

        if emergent:
            print(f"\n  EmergentDB speedups:")
            if lancedb_flat:
                print(f"    vs LanceDB (flat): {lancedb_flat.search_time_us/emergent.search_time_us:.1f}x faster")
            if chromadb_default:
                print(f"    vs ChromaDB (hnsw): {chromadb_default.search_time_us/emergent.search_time_us:.1f}x faster")


def main():
    print("=" * 70)
    print("EmergentDB Comprehensive Benchmark")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Vector dimensions: {VECTOR_DIM}")
    print(f"  Scales: {SCALES}")
    print(f"  Queries per scale: {N_QUERIES}")
    print(f"  Top-K: {K}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for n in SCALES:
        results = run_scale_benchmark(n)
        all_results.extend(results)

    # Print summary
    print_summary(all_results)

    # Save results
    results_data = {
        "config": {
            "vector_dim": VECTOR_DIM,
            "scales": SCALES,
            "n_queries": N_QUERIES,
            "k": K
        },
        "results": [asdict(r) for r in all_results]
    }

    output_file = OUTPUT_DIR / "comprehensive_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
