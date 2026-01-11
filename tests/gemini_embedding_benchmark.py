#!/usr/bin/env python3
"""
Gemini Embedding Benchmark for EmergentDB
Generates real embeddings using Google's Gemini API and tests them against EmergentDB.

Based on: https://ai.google.dev/gemini-api/docs/embeddings
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Configuration
GEMINI_MODEL = "gemini-embedding-001"
OUTPUT_DIM = 768  # Recommended dimension for balance of quality/size
CHUNK_SIZE = 500  # Characters per chunk
OVERLAP = 100     # Overlap between chunks

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    # Split by paragraphs first
    paragraphs = text.split('\n\n')

    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 50]  # Filter out tiny chunks

def get_gemini_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    """Get embeddings from Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:batchEmbedContents"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    # Batch embed
    requests_data = []
    for text in texts:
        requests_data.append({
            "model": f"models/{GEMINI_MODEL}",
            "content": {
                "parts": [{"text": text}]
            },
            "outputDimensionality": OUTPUT_DIM,
            "taskType": "RETRIEVAL_DOCUMENT"
        })

    payload = {"requests": requests_data}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []

    result = response.json()
    embeddings = []
    for emb in result.get("embeddings", []):
        values = emb.get("values", [])
        # Normalize the embedding
        norm = np.linalg.norm(values)
        if norm > 0:
            values = (np.array(values) / norm).tolist()
        embeddings.append(values)

    return embeddings

def read_markdown_files(directory: Path) -> Dict[str, str]:
    """Read all markdown files from directory."""
    files = {}
    for md_file in directory.glob("*.md"):
        try:
            content = md_file.read_text(encoding='utf-8')
            files[md_file.name] = content
        except Exception as e:
            print(f"Error reading {md_file}: {e}")
    return files

def main():
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        print("Get your key from: https://aistudio.google.com/app/apikey")
        return

    project_root = Path(__file__).parent.parent

    # Read markdown files
    print("Reading markdown files...")
    md_files = read_markdown_files(project_root)

    if not md_files:
        print("No markdown files found!")
        return

    print(f"Found {len(md_files)} markdown files:")
    for name in md_files:
        print(f"  - {name}")

    # Chunk all texts
    print("\nChunking texts...")
    all_chunks = []
    chunk_sources = []

    for filename, content in md_files.items():
        chunks = chunk_text(content)
        print(f"  {filename}: {len(chunks)} chunks")
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_sources.append(filename)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Get embeddings in batches (Gemini allows up to 100 per request)
    print(f"\nGenerating {OUTPUT_DIM}-dim embeddings with {GEMINI_MODEL}...")

    batch_size = 50
    all_embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)...")

        embeddings = get_gemini_embeddings(batch, api_key)
        all_embeddings.extend(embeddings)

        # Rate limiting
        time.sleep(0.5)

    print(f"\nGenerated {len(all_embeddings)} embeddings")

    # Save to JSON for Rust benchmark
    output_data = {
        "model": GEMINI_MODEL,
        "dimension": OUTPUT_DIM,
        "task_type": "RETRIEVAL_DOCUMENT",
        "count": len(all_embeddings),
        "embeddings": []
    }

    for i, (chunk, source, embedding) in enumerate(zip(all_chunks, chunk_sources, all_embeddings)):
        output_data["embeddings"].append({
            "id": i,
            "source": source,
            "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
            "vector": embedding
        })

    output_path = project_root / "tests" / "gemini_embeddings.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved embeddings to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Quick similarity test
    print("\n--- Quick Similarity Test ---")
    if len(all_embeddings) >= 3:
        emb1 = np.array(all_embeddings[0])
        emb2 = np.array(all_embeddings[1])
        emb3 = np.array(all_embeddings[-1])

        sim_1_2 = np.dot(emb1, emb2)
        sim_1_3 = np.dot(emb1, emb3)

        print(f"Chunk 0 (from {chunk_sources[0]}): {all_chunks[0][:60]}...")
        print(f"Chunk 1 (from {chunk_sources[1]}): {all_chunks[1][:60]}...")
        print(f"Chunk {len(all_chunks)-1} (from {chunk_sources[-1]}): {all_chunks[-1][:60]}...")
        print(f"\nSimilarity (0 vs 1): {sim_1_2:.4f}")
        print(f"Similarity (0 vs last): {sim_1_3:.4f}")

    # Test EmergentDB with real embeddings
    print("\n--- Testing EmergentDB with Gemini Embeddings ---")
    try:
        test_emergentdb(all_embeddings, all_chunks)
    except Exception as e:
        print(f"EmergentDB test failed: {e}")
        print("Make sure the EmergentDB server is running: cargo run --release -p api-server")

def test_emergentdb(embeddings: List[List[float]], texts: List[str]):
    """Test EmergentDB with the generated embeddings."""
    base_url = "http://localhost:3000"

    # Check health
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            print("EmergentDB server not responding")
            return
    except:
        print("Could not connect to EmergentDB server")
        return

    print(f"Inserting {len(embeddings)} vectors into EmergentDB...")

    # Batch insert
    insert_data = {
        "vectors": [
            {"id": i, "vector": emb, "metadata": {"text": texts[i][:100]}}
            for i, emb in enumerate(embeddings)
        ]
    }

    start = time.time()
    resp = requests.post(f"{base_url}/vectors/batch_insert", json=insert_data)
    insert_time = time.time() - start

    if resp.status_code != 200:
        print(f"Insert failed: {resp.text}")
        return

    print(f"Inserted in {insert_time:.3f}s ({len(embeddings)/insert_time:.0f} vec/s)")

    # Evolve to find optimal config
    print("\nRunning evolution...")
    start = time.time()
    resp = requests.post(f"{base_url}/qd/evolve")
    evolve_time = time.time() - start

    if resp.status_code == 200:
        result = resp.json()
        print(f"Evolution completed in {evolve_time:.2f}s")
        print(f"Best config: {result.get('index_type', 'unknown')}")
        print(f"Fitness: {result.get('fitness', 0):.3f}")

    # Search test
    print("\nSearch benchmark...")
    query_embedding = embeddings[0]  # Use first embedding as query

    latencies = []
    for _ in range(100):
        start = time.time()
        resp = requests.post(f"{base_url}/vectors/search", json={
            "query": query_embedding,
            "k": 10
        })
        latencies.append((time.time() - start) * 1000)  # ms

    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)

    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"P99 latency: {p99_latency:.2f}ms")

    # Show results
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        print(f"\nTop 5 results for query: '{texts[0][:50]}...'")
        for r in results[:5]:
            idx = r.get("id", 0)
            score = r.get("score", 0)
            text_preview = texts[idx][:60] if idx < len(texts) else "N/A"
            print(f"  [{score:.4f}] {text_preview}...")

if __name__ == "__main__":
    main()
