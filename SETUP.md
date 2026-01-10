# EmergentDB Setup Guide

A comprehensive guide to get EmergentDB running locally with document ingestion and Gemini OCR.

## Prerequisites

Before starting, ensure you have the following installed:

| Tool | Version | Check Command |
|------|---------|---------------|
| Rust | 1.75+ | `rustc --version` |
| Cargo | 1.75+ | `cargo --version` |
| Python | 3.10+ | `python3 --version` |
| Bun (or npm) | Latest | `bun --version` |

### Install Prerequisites

**macOS (Homebrew):**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python 3
brew install python@3.11

# Install Bun (recommended for frontend)
curl -fsSL https://bun.sh/install | bash
```

**Linux (Ubuntu/Debian):**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python 3
sudo apt update && sudo apt install python3 python3-pip python3-venv

# Install Bun
curl -fsSL https://bun.sh/install | bash
```

---

## Quick Start (5 Minutes)

### 1. Clone and Enter Directory
```bash
cd emergentDB
```

### 2. Set Up Environment Variables
```bash
# Copy the example .env file (if not already present)
cp .env.example .env 2>/dev/null || true

# Ensure your .env has the Gemini API key:
# GEMINI_API_KEY=your_key_here
```

### 3. Build the Rust Backend
```bash
# Standard build
cargo build --release

# For Apple M1/M2/M3/M4 (recommended for best performance)
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release
```

### 4. Start the API Server
```bash
# Start with default settings (port 3000, 768 dimensions)
cargo run --release -p api-server

# Or with custom settings
PORT=8080 VECTOR_DIM=1536 cargo run --release -p api-server
```

### 5. Start the Frontend (New Terminal)
```bash
cd frontend
bun install
bun run dev
```

### 6. Open in Browser
- **Frontend Dashboard:** http://localhost:3000
- **API Server:** http://localhost:3001 (if using port 3001)

---

## Running the Document Ingestion CLI

The ingestion CLI uses Gemini for OCR and embeddings to process documents.

### 1. Set Up Python Environment
```bash
cd examples/ingestion
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the CLI
```bash
# Interactive mode
python ingest.py

# Ingest a single file
python ingest.py --file /path/to/document.pdf

# Ingest a directory
python ingest.py --dir /path/to/documents

# Query the database
python ingest.py --query "What is machine learning?"

# List all documents
python ingest.py --list
```

### CLI Commands Reference

| Command | Description |
|---------|-------------|
| `--file <path>` | Ingest a single document (PDF, image, text) |
| `--dir <path>` | Ingest all documents in a directory |
| `--query <text>` | Search the vector database |
| `--list` | List all ingested documents |
| `--delete <id>` | Delete a document by ID |
| `--clear` | Clear the entire database |
| `--stats` | Show database statistics |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        EmergentDB                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Frontend    │    │  API Server  │    │  Ingestion CLI   │   │
│  │  (Next.js)   │───▶│   (Axum)     │◀───│    (Python)      │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│                             │                     │              │
│                             ▼                     ▼              │
│                      ┌──────────────┐    ┌──────────────────┐   │
│                      │ Vector Core  │    │   Gemini API     │   │
│                      │ (Rust SIMD)  │    │  (OCR + Embed)   │   │
│                      └──────────────┘    └──────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    QD Engine (MAP-Elites)                 │   │
│  │  ┌────────────────┐          ┌────────────────────────┐  │   │
│  │  │   IndexQD      │          │      InsertQD          │  │   │
│  │  │ (Flat/HNSW/IVF)│          │ (SIMD Strategies)      │  │   │
│  │  └────────────────┘          └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints Reference

### Vector Operations

**Search Vectors**
```bash
curl -X POST http://localhost:3000/vectors/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, 0.2, ...], "k": 10}'
```

**Insert Vector**
```bash
curl -X POST http://localhost:3000/vectors/insert \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "vector": [0.1, 0.2, ...], "metadata": {"source": "doc1.pdf"}}'
```

**Batch Insert**
```bash
curl -X POST http://localhost:3000/vectors/batch_insert \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": 1, "vector": [...], "metadata": {...}}, ...]}'
```

### Document Ingestion (New)

**Ingest Document**
```bash
curl -X POST http://localhost:3000/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Ingest with OCR**
```bash
curl -X POST http://localhost:3000/ingest/ocr \
  -H "Content-Type: multipart/form-data" \
  -F "file=@scanned_document.pdf"
```

### QD Evolution

**Run Evolution**
```bash
curl -X POST http://localhost:3000/qd/evolve \
  -H "Content-Type: application/json" \
  -d '{"sample_size": 1000, "generations": 10}'
```

### System

**Health Check**
```bash
curl http://localhost:3000/health
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | API server port |
| `VECTOR_DIM` | `768` | Vector dimension (768 for Gemini, 1536 for OpenAI) |
| `GEMINI_API_KEY` | - | Gemini API key for OCR and embeddings |
| `ENVIRONMENT` | `development` | Environment mode |

### Rust Build Flags

```bash
# Apple Silicon optimization
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release

# Intel/AMD optimization
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Debug build (slower but with symbols)
cargo build
```

---

## Running Benchmarks

### Rust Benchmarks
```bash
# Scale benchmark (1K, 10K, 50K vectors)
cargo run --release --example scale_benchmark

# Index-specific benchmarks
cargo bench -p vector-core
```

### Python Comparison Benchmarks
```bash
cd tests
python3 -m venv venv
source venv/bin/activate
pip install numpy lancedb chromadb

python3 scale_comparison.py
```

Results are saved to `tests/benchmark_results/`.

---

## Troubleshooting

### Build Errors

**Error: `linker 'cc' not found`**
```bash
# macOS
xcode-select --install

# Linux
sudo apt install build-essential
```

**Error: `SIMD intrinsics not available`**
```bash
# Use portable SIMD fallback
RUSTFLAGS="" cargo build --release
```

### Runtime Errors

**Error: `Address already in use`**
```bash
# Find and kill process on port 3000
lsof -i :3000
kill -9 <PID>
```

**Error: `GEMINI_API_KEY not set`**
```bash
# Ensure .env file exists and is sourced
source .env
# Or export directly
export GEMINI_API_KEY=your_key_here
```

### Frontend Issues

**Error: `bun: command not found`**
```bash
# Install bun
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```

**Error: `Module not found`**
```bash
cd frontend
rm -rf node_modules bun.lock
bun install
```

---

## Development Workflow

### Making Changes

1. **Backend (Rust)**
   ```bash
   cargo watch -x "run --release -p api-server"  # Auto-reload on changes
   ```

2. **Frontend (Next.js)**
   ```bash
   cd frontend && bun run dev  # Hot reload enabled
   ```

3. **Ingestion CLI**
   ```bash
   cd examples/ingestion
   python ingest.py --help
   ```

### Running Tests
```bash
# All Rust tests
cargo test --workspace

# Specific crate tests
cargo test -p vector-core
cargo test -p qd-engine
```

---

## Performance Tips

1. **Use Release Builds**: Always use `--release` for production
2. **CPU Optimization**: Use `-C target-cpu=native` for best SIMD performance
3. **Vector Dimensions**: Match embedding model dimensions exactly
4. **Batch Operations**: Use batch insert for large datasets
5. **Let Evolution Run**: Allow the QD engine to find optimal configurations

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-repo/emergentdb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/emergentdb/discussions)

---

## Next Steps

1. Try the [scale benchmark](/examples/scale_benchmark.rs) to see performance
2. Use the [ingestion CLI](/examples/ingestion/) to add documents
3. Explore the [frontend dashboard](http://localhost:3000) for visualization
4. Read about [MAP-Elites algorithm](https://arxiv.org/abs/1504.04909) for understanding the QD engine
