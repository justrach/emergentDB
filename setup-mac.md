# EmergentDB Setup Guide for macOS

A step-by-step guide to get EmergentDB running on your Mac, including all required tools and dependencies.

## Required Tools

| Tool | Version | Purpose | Install Command |
|------|---------|---------|-----------------|
| **Xcode Command Line Tools** | Latest | C/C++ compiler, linker | `xcode-select --install` |
| **Rust** | 1.75+ | Core language | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Cargo** | 1.75+ | Rust package manager | Installed with Rust |
| **Bun** | Latest | Fast JS runtime (frontend) | `curl -fsSL https://bun.sh/install \| bash` |
| **Python** | 3.10+ | Ingestion CLI | `brew install python@3.11` |
| **Git** | 2.x | Version control | Pre-installed on macOS |

### Optional Tools

| Tool | Purpose | Install Command |
|------|---------|-----------------|
| **Homebrew** | Package manager | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` |
| **cargo-watch** | Auto-reload on changes | `cargo install cargo-watch` |
| **jq** | JSON processing | `brew install jq` |

## Step 1: Install Prerequisites

### 1.1 Install Xcode Command Line Tools

This provides the C/C++ compiler needed for some Rust dependencies.

```bash
xcode-select --install
```

A dialog will appear. Click "Install" and wait for completion.

### 1.2 Install Rust

```bash
# Download and run the Rust installer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the prompts (default installation is fine)
# Then reload your shell configuration
source ~/.cargo/env

# Verify installation
rustc --version   # Should show 1.75.0 or higher
cargo --version   # Should show 1.75.0 or higher
```

### 1.3 Install Bun (for frontend)

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash

# Reload shell
source ~/.bashrc  # or ~/.zshrc

# Verify
bun --version
```

### 1.4 Install Python (for ingestion CLI)

```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org

# Verify
python3 --version  # Should show 3.10+
```

## Step 2: Clone and Build

### 2.1 Clone the Repository

```bash
git clone https://github.com/your-repo/emergentdb.git
cd emergentdb
```

### 2.2 Build the Rust Backend

**For Apple Silicon (M1/M2/M3/M4) - Recommended:**

```bash
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release
```

This enables ARM NEON SIMD optimizations for best performance.

**For Intel Macs:**

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Standard build (works on all Macs):**

```bash
cargo build --release
```

### 2.3 Verify Build

```bash
# Run tests to ensure everything works
cargo test --workspace
```

You should see output like:
```
test result: ok. 73 passed; 0 failed; 0 ignored
```

## Step 3: Run the API Server

```bash
# Default configuration (port 3000, 768-dim vectors)
cargo run --release -p api-server
```

**With custom settings:**

```bash
# Change port and vector dimension
PORT=8080 VECTOR_DIM=1536 cargo run --release -p api-server
```

**Verify the server is running:**

```bash
curl http://localhost:3000/health
```

Expected response: `{"status":"ok"}`

## Step 4: Set Up the Frontend

Open a new terminal window:

```bash
cd frontend
bun install
bun run dev
```

Open your browser to: http://localhost:3000

## Step 5: Set Up the Ingestion CLI (Optional)

If you want to ingest documents with Gemini OCR:

```bash
cd examples/ingestion

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Set up Gemini API key:**

```bash
# Create .env file in project root
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Or export directly
export GEMINI_API_KEY=your_api_key_here
```

**Test the CLI:**

```bash
python ingest.py --help
```

## Quick Start Commands Summary

```bash
# 1. Install tools (one-time)
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
curl -fsSL https://bun.sh/install | bash

# 2. Build (from project root)
cd emergentdb
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release

# 3. Run tests
cargo test --workspace

# 4. Start API server (Terminal 1)
cargo run --release -p api-server

# 5. Start frontend (Terminal 2)
cd frontend && bun install && bun run dev
```

## Running Benchmarks

### Rust Benchmarks

```bash
# Criterion micro-benchmarks
cargo bench -p vector-core

# Scale benchmark (1K, 10K, 50K vectors)
cargo run --release --example scale_benchmark

# PDF document simulation
cargo run --release --example pdf_benchmark
```

### Python Comparison Benchmarks

```bash
cd tests

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install comparison databases
pip install numpy lancedb chromadb

# Run comparison
python3 scale_comparison.py
```

Results are saved to `tests/benchmark_results/`.

## API Usage Examples

### Insert a Vector

```bash
curl -X POST http://localhost:3000/vectors/insert \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "vector": [0.1, 0.2, 0.3, ...],  # 768 dimensions
    "metadata": {"source": "test.pdf"}
  }'
```

### Search Vectors

```bash
curl -X POST http://localhost:3000/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, ...],  # 768 dimensions
    "k": 10
  }'
```

### Run Evolution

```bash
curl -X POST http://localhost:3000/qd/evolve \
  -H "Content-Type: application/json" \
  -d '{"sample_size": 1000, "generations": 10}'
```

## Troubleshooting

### "linker 'cc' not found"

```bash
# Install Xcode Command Line Tools
xcode-select --install
```

### "Address already in use" (port 3000)

```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### Slow build times

```bash
# Use release profile for faster runtime (slower build)
cargo build --release

# Use debug for faster builds (slower runtime)
cargo build
```

### "bun: command not found"

```bash
# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc

# Or specify full path
~/.bun/bin/bun --version
```

### SIMD not detected

If performance is lower than expected on Apple Silicon:

```bash
# Verify you're using the optimized build
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release

# Check CPU features
sysctl -a | grep machdep.cpu.features
```

### Python version issues

```bash
# Check Python version
python3 --version

# Use specific version if multiple installed
/usr/local/bin/python3.11 --version

# Create venv with specific version
/usr/local/bin/python3.11 -m venv venv
```

## Development Workflow

### Auto-reload on Changes

```bash
# Install cargo-watch
cargo install cargo-watch

# Run with auto-reload
cargo watch -x "run --release -p api-server"
```

### Running Specific Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p vector-core
cargo test -p qd-engine

# Specific test
cargo test test_hnsw_search

# With output
cargo test -- --nocapture
```

### Code Formatting

```bash
cargo fmt        # Format code
cargo clippy     # Lint code
```

## Performance Tips

1. **Always use `--release`** for production workloads
2. **Use CPU-specific flags** for best SIMD performance:
   - Apple Silicon: `RUSTFLAGS="-C target-cpu=apple-m1"`
   - Intel: `RUSTFLAGS="-C target-cpu=native"`
3. **Match vector dimensions** to your embedding model exactly
4. **Use batch insert** for large datasets
5. **Let evolution complete** - the QD engine needs time to find optimal configurations

## Useful Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | API server port |
| `VECTOR_DIM` | 768 | Vector dimension |
| `GEMINI_API_KEY` | - | Gemini API key (for ingestion) |
| `RUST_LOG` | - | Logging level (debug, info, warn, error) |

## Getting Help

- **Run tests**: `cargo test --workspace`
- **Check API health**: `curl http://localhost:3000/health`
- **View logs**: `RUST_LOG=debug cargo run --release -p api-server`
- **GitHub Issues**: Report bugs and request features
