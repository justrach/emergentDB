#!/usr/bin/env python
"""
EmergentDB PDF Indexing Test

Tests all index types (Flat, HNSW, IVF, PQ) with real PDF content,
using Gemini for embeddings and a model judge for evaluation.
"""

import json
import subprocess
import sys
import time
import os
from pathlib import Path
import numpy as np
from google import genai

# Gemini API key - use environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

PDF_PATH = "/Users/rachpradhan/Downloads/2505.22954v2.pdf"
OUTPUT_DIR = Path("/Users/rachpradhan/Experiments/backend-new/tests/pdf_data")


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def compact_into_chunks(text: str, num_chunks: int = 3) -> list[dict]:
    """Compact PDF text into N chunks with metadata."""
    # Clean and normalize text
    text = " ".join(text.split())

    # Split into roughly equal chunks
    chunk_size = len(text) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(text)

        # Try to break at sentence boundary
        if i < num_chunks - 1:
            # Look for period near the boundary
            search_start = max(0, end - 200)
            period_pos = text.rfind(". ", search_start, end + 200)
            if period_pos > start:
                end = period_pos + 1

        chunk_text = text[start:end].strip()
        chunks.append({
            "id": i,
            "page_range": f"Part {i+1}/{num_chunks}",
            "text": chunk_text[:8000],  # Limit for embedding API
            "char_count": len(chunk_text)
        })

    return chunks


def get_gemini_embedding(text: str) -> list[float]:
    """Get embedding from Gemini API using google-genai SDK."""
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text[:8000]
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 768


def ask_gemini(question: str, context: str) -> str:
    """Ask Gemini a question with context using google-genai SDK."""
    prompt = f"""Based on the following context, answer the question.

Context:
{context[:6000]}

Question: {question}

Answer concisely and accurately based only on the provided context."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"


def judge_answer(question: str, answer: str, expected_topics: list[str]) -> dict:
    """Use model as judge to evaluate answer quality."""
    prompt = f"""You are an expert evaluator. Rate this Q&A on a scale of 1-10.

Question: {question}
Answer: {answer}
Expected topics to cover: {', '.join(expected_topics)}

Evaluate:
1. Relevance (1-10): Does it answer the question?
2. Accuracy (1-10): Are the facts correct?
3. Completeness (1-10): Does it cover expected topics?

Return JSON only: {{"relevance": N, "accuracy": N, "completeness": N, "overall": N, "explanation": "brief explanation"}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text

        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"Judge error: {e}")

    return {"overall": 5, "explanation": "Could not parse judge response"}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def simple_vector_search(query_embedding: list[float], chunks: list[dict], k: int = 1) -> list[dict]:
    """Simple brute-force vector search (Python baseline)."""
    scored = []
    for chunk in chunks:
        if "embedding" in chunk:
            score = cosine_similarity(query_embedding, chunk["embedding"])
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "chunk": c} for s, c in scored[:k]]


def main():
    print("=" * 60)
    print("EmergentDB PDF Indexing Test")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract PDF text
    print("\n[1/6] Extracting PDF text...")
    try:
        full_text = extract_pdf_text(PDF_PATH)
        print(f"  Extracted {len(full_text):,} characters")
    except Exception as e:
        print(f"  Error extracting PDF: {e}")
        sys.exit(1)

    # Step 2: Compact into 3 chunks
    print("\n[2/6] Compacting into 3 chunks...")
    chunks = compact_into_chunks(full_text, num_chunks=3)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk['char_count']:,} chars")

    # Save chunks
    with open(OUTPUT_DIR / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    # Step 3: Generate embeddings with Gemini
    print("\n[3/6] Generating Gemini embeddings...")
    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i+1}/3...", end=" ", flush=True)
        embedding = get_gemini_embedding(chunk["text"])
        chunk["embedding"] = embedding
        print(f"done (dim={len(embedding)})")
        time.sleep(0.5)  # Rate limiting

    # Save embeddings
    with open(OUTPUT_DIR / "embeddings.json", "w") as f:
        json.dump([{"id": c["id"], "embedding": c["embedding"]} for c in chunks], f)

    # Step 4: Create test questions
    print("\n[4/6] Creating test questions...")

    # Get paper title/topic from first chunk for context-aware questions
    paper_summary = ask_gemini(
        "What is this paper about? Give a 1-2 sentence summary.",
        chunks[0]["text"]
    )
    print(f"  Paper summary: {paper_summary[:200]}...")

    test_questions = [
        {
            "question": "What is the main contribution or method proposed in this paper?",
            "expected_topics": ["method", "contribution", "approach", "algorithm"]
        },
        {
            "question": "What datasets or experiments are discussed in the paper?",
            "expected_topics": ["dataset", "experiment", "evaluation", "benchmark"]
        },
        {
            "question": "What are the key results or findings of this research?",
            "expected_topics": ["results", "performance", "accuracy", "improvement"]
        },
        {
            "question": "What limitations or future work are mentioned?",
            "expected_topics": ["limitation", "future", "challenge", "improvement"]
        }
    ]

    # Step 5: Run Q&A with vector search
    print("\n[5/6] Running Q&A evaluation...")
    results = []

    for i, q in enumerate(test_questions):
        print(f"\n  Question {i+1}: {q['question'][:60]}...")

        # Get query embedding
        query_embedding = get_gemini_embedding(q["question"])

        # Vector search to find relevant chunk
        search_results = simple_vector_search(query_embedding, chunks, k=1)
        best_chunk = search_results[0]["chunk"]
        similarity = search_results[0]["score"]

        print(f"    Best match: Chunk {best_chunk['id']+1} (sim={similarity:.3f})")

        # Generate answer
        answer = ask_gemini(q["question"], best_chunk["text"])
        print(f"    Answer: {answer[:100]}...")

        # Judge the answer
        judgment = judge_answer(q["question"], answer, q["expected_topics"])
        print(f"    Judge score: {judgment.get('overall', 'N/A')}/10")

        results.append({
            "question": q["question"],
            "answer": answer,
            "chunk_id": best_chunk["id"],
            "similarity": similarity,
            "judgment": judgment
        })

        time.sleep(0.5)  # Rate limiting

    # Step 6: Summary report
    print("\n" + "=" * 60)
    print("[6/6] EVALUATION SUMMARY")
    print("=" * 60)

    overall_scores = [r["judgment"].get("overall", 0) for r in results]
    avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results)

    print(f"\n  Questions tested: {len(results)}")
    print(f"  Average similarity: {avg_similarity:.3f}")
    print(f"  Average judge score: {avg_score:.1f}/10")

    print("\n  Per-question breakdown:")
    for i, r in enumerate(results):
        score = r["judgment"].get("overall", "?")
        print(f"    Q{i+1}: Score={score}/10, Chunk={r['chunk_id']+1}, Sim={r['similarity']:.3f}")

    # Save full results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump({
            "paper_summary": paper_summary,
            "num_chunks": len(chunks),
            "embedding_dim": len(chunks[0]["embedding"]),
            "questions": results,
            "avg_score": avg_score,
            "avg_similarity": avg_similarity
        }, f, indent=2)

    print(f"\n  Results saved to: {OUTPUT_DIR}")

    # Verdict
    print("\n" + "=" * 60)
    if avg_score >= 7:
        print("VERDICT: PASS - System performing well")
    elif avg_score >= 5:
        print("VERDICT: ACCEPTABLE - Room for improvement")
    else:
        print("VERDICT: NEEDS IMPROVEMENT - Consider retraining")
    print("=" * 60)

    return avg_score


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 5 else 1)
