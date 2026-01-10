# EmergentDB - AI Agent Integration Guide

This guide covers how to integrate EmergentDB with AI agents, LLM applications, and automated workflows.

## Overview

EmergentDB provides a REST API designed for AI agent integration, supporting:
- Vector storage and retrieval for RAG (Retrieval Augmented Generation)
- Document ingestion with automatic OCR and embedding
- Tool call interface compatible with OpenAI and Anthropic formats
- Semantic search for context retrieval

## Quick Setup for Agents

### 1. Start the Server

```bash
# With persistence (recommended for agents)
DATA_DIR=./agent_data cargo run --release -p api-server

# Environment variables
PORT=3000          # API port
VECTOR_DIM=768     # Gemini embedding dimension (or 1536 for OpenAI)
DATA_DIR=./data    # Persistence directory
```

### 2. Health Check

```bash
curl http://localhost:3000/health
```

Response:
```json
{
  "status": "healthy",
  "vectors_count": 1234,
  "dimension": 768,
  "persistence_enabled": true
}
```

## API Endpoints for Agents

### Vector Operations

#### Insert Vector
```bash
POST /vectors/insert
Content-Type: application/json

{
  "id": 1,
  "vector": [0.1, 0.2, ...],  # 768 floats for Gemini
  "metadata": {"source": "document.pdf", "chunk": 0}
}
```

#### Batch Insert
```bash
POST /vectors/batch_insert
Content-Type: application/json

{
  "vectors": [
    {"id": 1, "vector": [...], "metadata": {...}},
    {"id": 2, "vector": [...], "metadata": {...}}
  ]
}
```

#### Semantic Search
```bash
POST /vectors/search
Content-Type: application/json

{
  "query": [0.1, 0.2, ...],  # Query embedding
  "k": 5                      # Number of results
}
```

Response:
```json
{
  "results": [
    {"id": 42, "score": 0.92},
    {"id": 17, "score": 0.85}
  ],
  "latency_ms": 0
}
```

### Tool Call Interface

EmergentDB provides a unified tool interface for LLM agents:

```bash
POST /tools/call
Content-Type: application/json

{
  "tool": "vector_search",
  "parameters": {
    "query": [0.1, 0.2, ...],
    "k": 5
  }
}
```

#### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `vector_search` | Semantic similarity search | `query`, `k`, `metric` |
| `graph_query` | Query context graph | `start_id`, `relation_types`, `max_depth` |
| `research` | Deep research on topic | `query`, `tier`, `max_sources` |
| `qd_diversify` | Generate diverse variations | `base_queries`, `n` |

### List Tools (for Agent Discovery)

```bash
GET /tools/list
```

Returns OpenAI-compatible tool definitions that agents can use for function calling.

## Python Agent Integration

### Using with LangChain

```python
import requests
from langchain.tools import Tool

def emergent_search(query_embedding: list, k: int = 5) -> list:
    """Search EmergentDB for similar vectors."""
    response = requests.post(
        "http://localhost:3000/vectors/search",
        json={"query": query_embedding, "k": k}
    )
    return response.json()["results"]

# Create LangChain tool
emergent_tool = Tool(
    name="semantic_search",
    description="Search for semantically similar documents",
    func=lambda q: emergent_search(get_embedding(q))
)
```

### Using with OpenAI Function Calling

```python
from openai import OpenAI
import requests

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}]

def handle_tool_call(query: str):
    # Get embedding from OpenAI
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Search EmergentDB
    results = requests.post(
        "http://localhost:3000/vectors/search",
        json={"query": embedding, "k": 5}
    ).json()

    return results["results"]
```

### Using with Anthropic Claude

```python
from anthropic import Anthropic
import requests

client = Anthropic()

tools = [{
    "name": "search_documents",
    "description": "Search indexed documents for relevant context",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}]

# Use with Claude's tool_use capability
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=tools,
    messages=[{"role": "user", "content": "Find documents about machine learning"}]
)
```

## RAG Pipeline Example

Complete example of a RAG pipeline using EmergentDB:

```python
from google import genai
from google.genai import types
import requests

# Initialize Gemini client
client = genai.Client(api_key="YOUR_API_KEY")

class EmergentRAG:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
        """Get embedding from Gemini."""
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )
        return list(result.embeddings[0].values)

    def index_document(self, doc_id: int, text: str, metadata: dict = None):
        """Index a document."""
        embedding = self.embed(text, "RETRIEVAL_DOCUMENT")
        requests.post(
            f"{self.base_url}/vectors/insert",
            json={"id": doc_id, "vector": embedding, "metadata": metadata}
        )

    def search(self, query: str, k: int = 5) -> list:
        """Search for relevant documents."""
        query_embedding = self.embed(query, "RETRIEVAL_QUERY")
        response = requests.post(
            f"{self.base_url}/vectors/search",
            json={"query": query_embedding, "k": k}
        )
        return response.json()["results"]

    def generate_with_context(self, query: str, k: int = 3) -> str:
        """Generate response with retrieved context."""
        # Retrieve relevant documents
        results = self.search(query, k)

        # Build context (you'd fetch actual text from your document store)
        context = f"Found {len(results)} relevant documents with scores: "
        context += ", ".join([f"ID {r['id']}: {r['score']:.2f}" for r in results])

        # Generate with Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return response.text

# Usage
rag = EmergentRAG()

# Index documents
rag.index_document(1, "Machine learning is a subset of AI...", {"source": "ml_intro.txt"})
rag.index_document(2, "Neural networks are inspired by the brain...", {"source": "nn.txt"})

# Search and generate
answer = rag.generate_with_context("What is machine learning?")
```

## Ingestion CLI for Agents

The Python ingestion CLI can be used programmatically by agents:

```python
import subprocess

# Ingest a document
subprocess.run([
    "python", "examples/ingestion/ingest.py",
    "ingest", "/path/to/document.pdf"
])

# Query the database
result = subprocess.run([
    "python", "examples/ingestion/ingest.py",
    "query", "What is machine learning?", "--k", "5"
], capture_output=True, text=True)

print(result.stdout)
```

## Best Practices for Agents

### 1. Use Batch Operations
For indexing multiple documents, use batch insert to reduce API calls:

```python
vectors = [
    {"id": i, "vector": embed(doc), "metadata": {"source": f"doc_{i}"}}
    for i, doc in enumerate(documents)
]
requests.post(f"{base_url}/vectors/batch_insert", json={"vectors": vectors})
```

### 2. Enable Persistence
Always use `DATA_DIR` for agent workloads to prevent data loss:

```bash
DATA_DIR=./agent_data cargo run --release -p api-server
```

### 3. Match Embedding Dimensions
Ensure your embedding model dimension matches `VECTOR_DIM`:

| Model | Dimension |
|-------|-----------|
| Gemini embedding-001 | 768 |
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |

### 4. Use Appropriate Task Types
For Gemini embeddings, use the correct task type:
- `RETRIEVAL_DOCUMENT` for indexing
- `RETRIEVAL_QUERY` for searching

### 5. Handle Rate Limits
Implement exponential backoff for embedding API calls:

```python
import time

def embed_with_retry(text, max_retries=3):
    for i in range(max_retries):
        try:
            return client.models.embed_content(...)
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)
```

## Error Handling

EmergentDB returns standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Not found |
| 500 | Server error |

Example error handling:

```python
response = requests.post(f"{base_url}/vectors/search", json=payload)
if response.status_code != 200:
    error = response.json().get("error", "Unknown error")
    raise Exception(f"EmergentDB error: {error}")
```

## Performance Tips

1. **Search Latency**: EmergentDB searches complete in <1ms for most workloads
2. **Batch Size**: Optimal batch size is 100-1000 vectors per request
3. **Connection Pooling**: Use `requests.Session()` for connection reuse
4. **Local Deployment**: Run EmergentDB on the same machine as your agent for lowest latency

## Next Steps

- See [SETUP.md](./SETUP.md) for detailed installation
- See [claude.md](./claude.md) for Claude Code integration
- See [gemini.md](./gemini.md) for Gemini API examples
