import { BenchmarkRunner } from "@/components/BenchmarkRunner";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { StatCard } from "@/components/StatCard";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#09090B]">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#09090B]/80 backdrop-blur-lg border-b border-[#27272a]">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#298DFF] flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="font-semibold text-lg">EmergentDB</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#benchmark" className="text-sm text-zinc-400 hover:text-white transition-colors">
              Benchmark
            </a>
            <a href="#how-it-works" className="text-sm text-zinc-400 hover:text-white transition-colors">
              How it Works
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#298DFF]/10 border border-[#298DFF]/30 mb-8">
            <span className="w-2 h-2 rounded-full bg-[#298DFF] animate-pulse" />
            <span className="text-sm text-[#298DFF] font-medium">Self-Optimizing Vector Database</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold leading-tight mb-6">
            <span className="text-white">Vector Search</span>
            <br />
            <span className="text-[#298DFF]">That Evolves</span>
          </h1>

          <p className="text-xl text-zinc-400 max-w-2xl mx-auto mb-12">
            EmergentDB uses MAP-Elites evolutionary algorithm to automatically discover
            the optimal index configuration for your workload.{" "}
            <span className="text-white font-medium">44-193x faster than LanceDB.</span>
          </p>

          {/* Key stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto">
            <StatCard
              title="Search Latency"
              value="42.9μs"
              subtitle="at 1K vectors"
              highlight
            />
            <StatCard
              title="vs LanceDB"
              value="193x"
              subtitle="faster at 1K"
              trend="up"
              trendValue="faster"
            />
            <StatCard
              title="vs ChromaDB"
              value="25x"
              subtitle="faster at 1K"
              trend="up"
              trendValue="faster"
            />
            <StatCard
              title="Recall@10"
              value="99%+"
              subtitle="accuracy floor"
            />
          </div>
        </div>
      </section>

      {/* Why EmergentDB Section */}
      <section className="py-20 px-6 bg-[#0e0e13]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <span className="mono-label text-[#298DFF]">The Problem</span>
            <h2 className="text-3xl md:text-4xl font-bold mt-4 mb-6">
              Big Vectors Are Slow
            </h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              As embedding dimensions grow (768-3072), traditional vector databases struggle.
              HNSW builds are slow. Flat search doesn&apos;t scale. You&apos;re left guessing which index works best.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-[#141419] rounded-xl p-6 border border-[#27272a]">
              <div className="w-12 h-12 rounded-lg bg-red-500/10 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Manual Tuning Hell</h3>
              <p className="text-sm text-zinc-400">
                HNSW M=16? M=32? ef_construction=100? Most teams guess and hope for the best.
              </p>
            </div>

            <div className="bg-[#141419] rounded-xl p-6 border border-[#27272a]">
              <div className="w-12 h-12 rounded-lg bg-yellow-500/10 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Workload Mismatch</h3>
              <p className="text-sm text-zinc-400">
                Optimal config for 1K vectors ≠ optimal for 100K. Databases don&apos;t adapt.
              </p>
            </div>

            <div className="bg-[#141419] rounded-xl p-6 border border-[#27272a]">
              <div className="w-12 h-12 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Recall vs Speed</h3>
              <p className="text-sm text-zinc-400">
                Fast search often means lower recall. You shouldn&apos;t have to choose.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <span className="mono-label text-[#298DFF]">The Solution</span>
            <h2 className="text-3xl md:text-4xl font-bold mt-4">
              Dual Quality-Diversity System
            </h2>
          </div>

          <ArchitectureDiagram />

          <div className="grid md:grid-cols-2 gap-8 mt-12">
            <div className="bg-[#0e0e13] rounded-xl p-6 border border-[#27272a]">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-[#298DFF]/20 flex items-center justify-center">
                  <span className="text-[#298DFF] font-bold">QD₁</span>
                </div>
                <h3 className="font-semibold text-lg">IndexQD</h3>
              </div>
              <p className="text-sm text-zinc-400 mb-4">
                3D behavior space: Recall × Latency × Memory. Evolves the optimal index type
                (HNSW, Flat, IVF) and hyperparameters for your data distribution.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-2 py-1 text-xs bg-[#141419] rounded border border-[#27272a]">HNSW M=16-64</span>
                <span className="px-2 py-1 text-xs bg-[#141419] rounded border border-[#27272a]">ef_construction=50-400</span>
                <span className="px-2 py-1 text-xs bg-[#141419] rounded border border-[#27272a]">ef_search=10-200</span>
              </div>
            </div>

            <div className="bg-[#0e0e13] rounded-xl p-6 border border-[#27272a]">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-[#298DFF]/20 flex items-center justify-center">
                  <span className="text-[#298DFF] font-bold">QD₂</span>
                </div>
                <h3 className="font-semibold text-lg">InsertQD</h3>
              </div>
              <p className="text-sm text-zinc-400 mb-4">
                2D behavior space: Throughput × Efficiency. Discovers the fastest SIMD
                insertion strategy for your hardware and batch sizes.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-2 py-1 text-xs bg-[#141419] rounded border border-[#27272a]">SIMD-Unrolled</span>
                <span className="px-2 py-1 text-xs bg-[#141419] rounded border border-[#27272a]">SIMD-Chunked</span>
                <span className="px-2 py-1 text-xs bg-[#141419] rounded border border-[#27272a]">5.6M vec/s</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Benchmark Section */}
      <section id="benchmark" className="py-20 px-6 bg-[#0e0e13]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <span className="mono-label text-[#298DFF]">Live Benchmark</span>
            <h2 className="text-3xl md:text-4xl font-bold mt-4 mb-4">
              Real Performance Comparison
            </h2>
            <p className="text-zinc-400 max-w-xl mx-auto">
              768-dimensional vectors (OpenAI text-embedding-3-small size).
              All benchmarks run on the same hardware with 99%+ recall.
            </p>
          </div>

          <BenchmarkRunner />
        </div>
      </section>

      {/* Code Example */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <span className="mono-label text-[#298DFF]">Quick Start</span>
            <h2 className="text-3xl md:text-4xl font-bold mt-4">
              Simple API
            </h2>
          </div>

          <div className="bg-[#0e0e13] rounded-xl overflow-hidden border border-[#27272a]">
            <div className="flex items-center gap-2 px-4 py-3 bg-[#141419] border-b border-[#27272a]">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="ml-3 text-xs text-zinc-500 font-mono">example.rs</span>
            </div>
            <pre className="p-6 overflow-x-auto text-sm">
              <code className="font-mono text-zinc-300">
{`use vector_core::index::emergent::{EmergentConfig, EmergentIndex};

// Create with search-optimized preset
let config = EmergentConfig::search_first();
let mut index = EmergentIndex::new(config);

// Insert your vectors
for (id, embedding) in vectors {
    index.insert(id, embedding)?;
}

// Evolve to find optimal configuration
let elite = index.evolve()?;
println!("Selected: {} (fitness: {:.3})",
    elite.genome.index_type, elite.fitness);

// Search - now 44-193x faster than LanceDB
let results = index.search(&query, 10)?;`}
              </code>
            </pre>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-[#27272a]">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#298DFF] flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="font-semibold">EmergentDB</span>
          </div>
          <p className="text-sm text-zinc-500">
            Built with Rust, powered by MAP-Elites evolution
          </p>
          <div className="flex items-center gap-4">
            <a href="#" className="text-sm text-zinc-400 hover:text-white">Documentation</a>
            <a href="#" className="text-sm text-zinc-400 hover:text-white">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
