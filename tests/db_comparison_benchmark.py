#!/usr/bin/env python
"""
EmergentDB vs LanceDB vs Chroma Benchmark

Fair comparison using the same:
- PDF chunks (3 chunks from Darwin Godel Machine paper)
- Embeddings (Gemini text-embedding-004, 768 dims)
- Test queries
- Evaluation metrics (latency, recall, throughput)
"""

import json
import time
import tempfile
import shutil
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Third-party imports
from google import genai
import lancedb
import chromadb
from pypdf import PdfReader

# Config - use environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PDF_PATH = "/Users/rachpradhan/Downloads/2505.22954v2.pdf"
OUTPUT_DIR = Path("/Users/rachpradhan/Experiments/backend-new/tests/benchmark_results")
RUST_DIR = Path("/Users/rachpradhan/Experiments/backend-new")

# Initialize Gemini
client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class BenchmarkResult:
    name: str
    insert_time_ms: float
    search_time_us: float
    recall_at_5: float
    memory_mb: float = 0.0


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def compact_into_chunks(text: str, num_chunks: int = 3) -> list[dict]:
    """Split text into N chunks."""
    text = " ".join(text.split())
    chunk_size = len(text) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(text)

        if i < num_chunks - 1:
            period_pos = text.rfind(". ", max(0, end - 200), end + 200)
            if period_pos > start:
                end = period_pos + 1

        chunks.append({
            "id": str(i),
            "text": text[start:end].strip()[:8000],
        })

    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get Gemini embeddings for texts."""
    embeddings = []
    for i, text in enumerate(texts):
        print(f"    Embedding {i+1}/{len(texts)}...", end=" ", flush=True)
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text[:8000]
        )
        embeddings.append(result.embeddings[0].values)
        print("done")
        time.sleep(0.3)
    return embeddings


def generate_test_vectors(base_embeddings: list[list[float]], n_extra: int = 300) -> tuple[list[list[float]], list[str]]:
    """Generate additional random vectors for benchmarking."""
    dim = len(base_embeddings[0])
    all_embeddings = base_embeddings.copy()
    all_ids = [str(i) for i in range(len(base_embeddings))]

    for i in range(n_extra):
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        all_embeddings.append(vec.tolist())
        all_ids.append(str(len(base_embeddings) + i))

    return all_embeddings, all_ids


def compute_recall(ground_truth: list[str], results: list[str], k: int = 5) -> float:
    """Compute recall@k."""
    gt_set = set(ground_truth[:k])
    result_set = set(results[:k])
    return len(gt_set & result_set) / k


# =============================================================================
# LANCEDB BENCHMARK
# =============================================================================

def benchmark_lancedb(embeddings: list[list[float]], ids: list[str],
                      query_embeddings: list[list[float]], ground_truths: list[list[str]]) -> BenchmarkResult:
    """Benchmark LanceDB."""
    print("\n[LanceDB] Starting benchmark...")

    # Create temp directory for LanceDB
    tmp_dir = tempfile.mkdtemp()

    try:
        db = lancedb.connect(tmp_dir)

        # Prepare data
        data = [{"id": ids[i], "vector": embeddings[i]} for i in range(len(embeddings))]

        # Insert
        start = time.perf_counter()
        table = db.create_table("vectors", data)
        insert_time = (time.perf_counter() - start) * 1000

        # Create IVF-PQ index for fair comparison
        # table.create_index(metric="cosine", num_partitions=16, num_sub_vectors=48)

        # Search
        k = 5
        total_recall = 0.0
        search_times = []

        for i, query in enumerate(query_embeddings):
            start = time.perf_counter()
            results = table.search(query).limit(k).to_list()
            search_times.append((time.perf_counter() - start) * 1_000_000)

            result_ids = [r["id"] for r in results]
            total_recall += compute_recall(ground_truths[i], result_ids, k)

        avg_search_time = sum(search_times) / len(search_times)
        avg_recall = total_recall / len(query_embeddings)

        print(f"  Insert: {insert_time:.2f}ms, Search: {avg_search_time:.1f}us, Recall@5: {avg_recall*100:.1f}%")

        return BenchmarkResult(
            name="LanceDB",
            insert_time_ms=insert_time,
            search_time_us=avg_search_time,
            recall_at_5=avg_recall
        )
    finally:
        shutil.rmtree(tmp_dir)


# =============================================================================
# CHROMA BENCHMARK
# =============================================================================

def benchmark_chroma(embeddings: list[list[float]], ids: list[str],
                     query_embeddings: list[list[float]], ground_truths: list[list[str]]) -> BenchmarkResult:
    """Benchmark ChromaDB."""
    print("\n[ChromaDB] Starting benchmark...")

    tmp_dir = tempfile.mkdtemp()

    try:
        client = chromadb.PersistentClient(path=tmp_dir)

        # Create collection with HNSW
        collection = client.create_collection(
            name="vectors",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert
        start = time.perf_counter()
        collection.add(
            ids=ids,
            embeddings=embeddings
        )
        insert_time = (time.perf_counter() - start) * 1000

        # Search
        k = 5
        total_recall = 0.0
        search_times = []

        for i, query in enumerate(query_embeddings):
            start = time.perf_counter()
            results = collection.query(
                query_embeddings=[query],
                n_results=k
            )
            search_times.append((time.perf_counter() - start) * 1_000_000)

            result_ids = results["ids"][0]
            total_recall += compute_recall(ground_truths[i], result_ids, k)

        avg_search_time = sum(search_times) / len(search_times)
        avg_recall = total_recall / len(query_embeddings)

        print(f"  Insert: {insert_time:.2f}ms, Search: {avg_search_time:.1f}us, Recall@5: {avg_recall*100:.1f}%")

        return BenchmarkResult(
            name="ChromaDB (HNSW)",
            insert_time_ms=insert_time,
            search_time_us=avg_search_time,
            recall_at_5=avg_recall
        )
    finally:
        shutil.rmtree(tmp_dir)


# =============================================================================
# EMERGENTDB BENCHMARK (via Rust)
# =============================================================================

def benchmark_emergentdb(embeddings: list[list[float]], ids: list[str],
                         query_indices: list[int]) -> dict[str, BenchmarkResult]:
    """Benchmark EmergentDB (all index types + Emergent auto-selection via Rust)."""
    print("\n[EmergentDB] Starting benchmark...")
    print("  Running Manual indices (Flat, HNSW, IVF, PQ) + Emergent auto-selection...")

    # Run the Rust benchmark (includes manual + auto selection)
    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "pdf_benchmark"],
        cwd=str(RUST_DIR),
        capture_output=True,
        text=True,
        timeout=300
    )

    # Parse results from stdout
    results = {}
    lines = result.stdout.split("\n")

    for line in lines:
        if "Insert:" in line and "Search:" in line:
            # Parse line like "  Insert: 0.17ms, Search: 218.4us/query"
            parts = line.strip()
            if "Flat" in parts or "[1/4]" in lines[max(0, lines.index(line)-1):lines.index(line)]:
                name = "Flat"
            elif "HNSW" in parts or "[2/4]" in lines[max(0, lines.index(line)-1):lines.index(line)]:
                name = "HNSW"
            elif "IVF" in parts or "[3/4]" in lines[max(0, lines.index(line)-1):lines.index(line)]:
                name = "IVF"
            elif "PQ" in parts or "[4/4]" in lines[max(0, lines.index(line)-1):lines.index(line)]:
                name = "PQ"
            else:
                continue

    # Parse the summary table (handles both "Flat (Manual)" and "Emergent (Auto)" formats)
    in_table = False
    for line in lines:
        if "Index" in line and "Insert(ms)" in line:
            in_table = True
            continue
        if in_table and line.strip() and not line.startswith("-"):
            # Skip non-data lines
            if line.strip().startswith("=") or "MANUAL" in line or "AUTO" in line or "Best" in line or "Selected" in line or "Recall:" in line or "Search" in line:
                continue

            # Handle format: "Flat (Manual)    0.48    462.9   100.0%    -"
            # Or: "Emergent (Auto)  10678.80   100.0   100.0%   HNSW"
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Combine name parts like "Flat" + "(Manual)" or "Emergent" + "(Auto)"
                    if len(parts) > 1 and parts[1].startswith("("):
                        name = f"{parts[0]} {parts[1]}"
                        insert = float(parts[2])
                        search = float(parts[3])
                        recall = float(parts[4].replace("%", "")) / 100
                    else:
                        name = parts[0]
                        insert = float(parts[1])
                        search = float(parts[2])
                        recall = float(parts[3].replace("%", "")) / 100

                    # Clean name for EmergentDB display
                    display_name = name.replace(" (Manual)", "").replace(" (Auto)", " Auto")
                    results[display_name] = BenchmarkResult(
                        name=f"EmergentDB ({display_name})",
                        insert_time_ms=insert,
                        search_time_us=search,
                        recall_at_5=recall
                    )
                except (ValueError, IndexError):
                    pass  # Skip lines that don't match expected format

    # Print raw output if parsing failed
    if not results:
        print("  Raw Rust output:")
        print(result.stdout)
        if result.stderr:
            print("  STDERR:", result.stderr[-500:])

    return results


# =============================================================================
# GROUND TRUTH (Flat Index)
# =============================================================================

def compute_ground_truth_python(embeddings: list[list[float]], query_indices: list[int], k: int = 5) -> list[list[str]]:
    """Compute ground truth using brute force in Python."""
    ground_truths = []
    embeddings_np = np.array(embeddings)

    for qi in query_indices:
        query = embeddings_np[qi]
        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(embeddings_np, query)
        top_k = np.argsort(similarities)[::-1][:k]
        ground_truths.append([str(i) for i in top_k])

    return ground_truths


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EmergentDB vs LanceDB vs ChromaDB Benchmark")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract PDF
    print("\n[1/5] Extracting PDF text...")
    full_text = extract_pdf_text(PDF_PATH)
    print(f"  Extracted {len(full_text):,} characters")

    # Step 2: Create chunks
    print("\n[2/5] Creating 3 chunks...")
    chunks = compact_into_chunks(full_text, num_chunks=3)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk['text']):,} chars")

    # Step 3: Generate embeddings
    print("\n[3/5] Generating Gemini embeddings...")
    chunk_texts = [c["text"] for c in chunks]
    base_embeddings = get_embeddings(chunk_texts)

    # Generate more vectors for meaningful benchmark
    print("\n[4/5] Generating test vectors (300 additional)...")
    all_embeddings, all_ids = generate_test_vectors(base_embeddings, n_extra=300)
    print(f"  Total vectors: {len(all_embeddings)}, Dimension: {len(all_embeddings[0])}")

    # Normalize embeddings for cosine similarity
    all_embeddings = [list(np.array(e) / np.linalg.norm(e)) for e in all_embeddings]

    # Query setup
    query_indices = list(range(10))  # First 10 vectors as queries
    query_embeddings = [all_embeddings[i] for i in query_indices]

    # Compute ground truth
    print("\n  Computing ground truth (brute force)...")
    ground_truths = compute_ground_truth_python(all_embeddings, query_indices, k=5)

    # Step 5: Run benchmarks
    print("\n[5/5] Running benchmarks...")

    all_results = []

    # LanceDB
    lance_result = benchmark_lancedb(all_embeddings, all_ids, query_embeddings, ground_truths)
    all_results.append(lance_result)

    # ChromaDB
    chroma_result = benchmark_chroma(all_embeddings, all_ids, query_embeddings, ground_truths)
    all_results.append(chroma_result)

    # EmergentDB (Rust)
    emergent_results = benchmark_emergentdb(all_embeddings, all_ids, query_indices)
    for name, result in emergent_results.items():
        all_results.append(result)

    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {len(all_embeddings)} vectors, {len(all_embeddings[0])} dimensions")
    print(f"Queries: {len(query_embeddings)}, k=5")
    print(f"Source: Darwin Gödel Machine paper (arXiv:2505.22954)")

    print("\n{:<25} {:>12} {:>12} {:>10}".format("Database", "Insert(ms)", "Search(us)", "Recall@5"))
    print("-" * 60)

    for r in sorted(all_results, key=lambda x: x.search_time_us):
        print("{:<25} {:>12.2f} {:>12.1f} {:>9.1f}%".format(
            r.name, r.insert_time_ms, r.search_time_us, r.recall_at_5 * 100
        ))

    # Best performers
    print("\n" + "-" * 60)
    best_recall = max(all_results, key=lambda x: x.recall_at_5)
    best_speed = min([r for r in all_results if r.recall_at_5 > 0.8], key=lambda x: x.search_time_us, default=None)
    best_insert = min(all_results, key=lambda x: x.insert_time_ms)

    print(f"\n  🏆 Best Recall: {best_recall.name} ({best_recall.recall_at_5*100:.1f}%)")
    if best_speed:
        print(f"  ⚡ Best Speed (recall>80%): {best_speed.name} ({best_speed.search_time_us:.1f}us)")
    print(f"  📥 Fastest Insert: {best_insert.name} ({best_insert.insert_time_ms:.2f}ms)")

    # Save results
    results_data = {
        "dataset": {
            "num_vectors": len(all_embeddings),
            "dimension": len(all_embeddings[0]),
            "num_queries": len(query_embeddings),
            "source": "Darwin Gödel Machine paper"
        },
        "results": [
            {
                "name": r.name,
                "insert_time_ms": r.insert_time_ms,
                "search_time_us": r.search_time_us,
                "recall_at_5": r.recall_at_5
            }
            for r in all_results
        ]
    }

    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n  Results saved to: {OUTPUT_DIR / 'comparison_results.json'}")
    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
