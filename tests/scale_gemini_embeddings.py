#!/usr/bin/env python3
"""
Scale up Gemini embeddings to test HNSW configurations.
Creates synthetic variations by adding small noise to real embeddings.
"""

import json
import numpy as np
from pathlib import Path

def main():
    # Load original embeddings
    input_path = Path(__file__).parent / "gemini_embeddings.json"
    with open(input_path) as f:
        data = json.load(f)

    original_embeddings = [e["vector"] for e in data["embeddings"]]
    original_texts = [e["text"] for e in data["embeddings"]]
    original_sources = [e["source"] for e in data["embeddings"]]

    n_original = len(original_embeddings)
    dim = len(original_embeddings[0])

    print(f"Original embeddings: {n_original}")
    print(f"Dimension: {dim}")

    # Target sizes to test HNSW configurations
    target_sizes = [1000, 5000, 10000]

    for target in target_sizes:
        print(f"\nGenerating {target} embeddings...")

        scaled_embeddings = []
        scaled_texts = []
        scaled_sources = []

        # Add noise to create variations while preserving semantic structure
        noise_scale = 0.05  # Small noise to maintain similarity

        for i in range(target):
            # Cycle through original embeddings
            orig_idx = i % n_original
            orig_emb = np.array(original_embeddings[orig_idx])

            if i < n_original:
                # Keep originals unchanged
                new_emb = orig_emb
            else:
                # Add small Gaussian noise
                noise = np.random.normal(0, noise_scale, dim).astype(np.float32)
                new_emb = orig_emb + noise

                # Re-normalize
                norm = np.linalg.norm(new_emb)
                if norm > 0:
                    new_emb = new_emb / norm

            scaled_embeddings.append(new_emb.tolist())
            scaled_texts.append(f"[var {i // n_original}] {original_texts[orig_idx][:100]}")
            scaled_sources.append(original_sources[orig_idx])

        # Save scaled embeddings
        output_data = {
            "model": data["model"],
            "dimension": dim,
            "count": target,
            "noise_scale": noise_scale,
            "original_count": n_original,
            "embeddings": [
                {
                    "id": i,
                    "source": scaled_sources[i],
                    "text": scaled_texts[i],
                    "vector": scaled_embeddings[i]
                }
                for i in range(target)
            ]
        }

        output_path = Path(__file__).parent / f"gemini_embeddings_{target}.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path.name} ({file_size_mb:.1f} MB)")

        # Verify similarity is preserved
        if len(scaled_embeddings) >= 2:
            e1 = np.array(scaled_embeddings[0])
            e2 = np.array(scaled_embeddings[1])
            e_var = np.array(scaled_embeddings[n_original]) if target > n_original else e2

            sim_adjacent = np.dot(e1, e2)
            sim_variation = np.dot(e1, e_var)

            print(f"  Similarity (original adjacent): {sim_adjacent:.4f}")
            print(f"  Similarity (original vs variation): {sim_variation:.4f}")

if __name__ == "__main__":
    main()
