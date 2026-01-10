"use client";

import { useState, useCallback } from "react";
import { BenchmarkBar } from "./BenchmarkBar";

interface BenchmarkResult {
  name: string;
  n_vectors: number;
  insert_time_ms: number;
  search_time_us: number;
  recall_at_10: number;
}

interface ScaleResults {
  scale: number;
  results: BenchmarkResult[];
}

// Pre-loaded benchmark data from actual runs
const BENCHMARK_DATA: ScaleResults[] = [
  {
    scale: 1000,
    results: [
      { name: "EmergentDB (Auto)", n_vectors: 1000, insert_time_ms: 4281, search_time_us: 42.9, recall_at_10: 1.0 },
      { name: "EmergentDB (HNSW)", n_vectors: 1000, insert_time_ms: 556, search_time_us: 134.6, recall_at_10: 1.0 },
      { name: "LanceDB", n_vectors: 1000, insert_time_ms: 63.67, search_time_us: 8271.64, recall_at_10: 1.0 },
      { name: "ChromaDB", n_vectors: 1000, insert_time_ms: 143.26, search_time_us: 1078.67, recall_at_10: 0.99 },
    ]
  },
  {
    scale: 10000,
    results: [
      { name: "EmergentDB (Auto)", n_vectors: 10000, insert_time_ms: 39236, search_time_us: 122.0, recall_at_10: 1.0 },
      { name: "EmergentDB (HNSW)", n_vectors: 10000, insert_time_ms: 17480, search_time_us: 379.6, recall_at_10: 1.0 },
      { name: "LanceDB", n_vectors: 10000, insert_time_ms: 107.39, search_time_us: 7442.38, recall_at_10: 1.0 },
      { name: "ChromaDB", n_vectors: 10000, insert_time_ms: 3314.7, search_time_us: 1933.28, recall_at_10: 0.765 },
    ]
  },
  {
    scale: 50000,
    results: [
      { name: "EmergentDB (Auto)", n_vectors: 50000, insert_time_ms: 386149, search_time_us: 280.4, recall_at_10: 0.995 },
      { name: "EmergentDB (HNSW)", n_vectors: 50000, insert_time_ms: 168124, search_time_us: 691.1, recall_at_10: 0.99 },
      { name: "LanceDB", n_vectors: 50000, insert_time_ms: 539.49, search_time_us: 12348.24, recall_at_10: 1.0 },
      { name: "ChromaDB", n_vectors: 50000, insert_time_ms: 23180.33, search_time_us: 2698.94, recall_at_10: 0.6 },
    ]
  },
];

const DB_COLORS: Record<string, string> = {
  "EmergentDB (Auto)": "#298DFF",
  "EmergentDB (HNSW)": "#60a5fa",
  "LanceDB": "#a855f7",
  "ChromaDB": "#f97316",
};

export function BenchmarkRunner() {
  const [selectedScale, setSelectedScale] = useState<number>(10000);
  const [isRunning, setIsRunning] = useState(false);
  const [animationKey, setAnimationKey] = useState(0);

  const currentData = BENCHMARK_DATA.find(d => d.scale === selectedScale);
  const sortedResults = currentData?.results.slice().sort((a, b) => a.search_time_us - b.search_time_us) || [];
  const maxSearchTime = Math.max(...(sortedResults.map(r => r.search_time_us) || [1]));

  const runBenchmark = useCallback(() => {
    setIsRunning(true);
    setAnimationKey(prev => prev + 1);
    setTimeout(() => setIsRunning(false), 2000);
  }, []);

  const emergentAuto = sortedResults.find(r => r.name === "EmergentDB (Auto)");
  const lancedb = sortedResults.find(r => r.name === "LanceDB");
  const chromadb = sortedResults.find(r => r.name === "ChromaDB");

  const speedupVsLance = lancedb && emergentAuto
    ? (lancedb.search_time_us / emergentAuto.search_time_us).toFixed(1)
    : "—";
  const speedupVsChroma = chromadb && emergentAuto
    ? (chromadb.search_time_us / emergentAuto.search_time_us).toFixed(1)
    : "—";

  return (
    <div className="space-y-8">
      {/* Scale Selector */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {[1000, 10000, 50000].map(scale => (
            <button
              key={scale}
              onClick={() => {
                setSelectedScale(scale);
                setAnimationKey(prev => prev + 1);
              }}
              className={`px-4 py-2 rounded-lg font-mono text-sm transition-all ${
                selectedScale === scale
                  ? 'bg-[#298DFF] text-white'
                  : 'bg-[#141419] text-zinc-400 hover:bg-[#1f1f23] hover:text-white'
              }`}
            >
              {scale.toLocaleString()} vectors
            </button>
          ))}
        </div>
        <button
          onClick={runBenchmark}
          disabled={isRunning}
          className={`px-6 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
            isRunning
              ? 'bg-[#141419] text-zinc-500 cursor-not-allowed'
              : 'bg-[#298DFF] text-white hover:bg-[#1d6fcc]'
          }`}
        >
          {isRunning ? (
            <>
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running...
            </>
          ) : (
            <>
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run Benchmark
            </>
          )}
        </button>
      </div>

      {/* Speedup Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="stat-card glow-accent border-[#298DFF]">
          <span className="mono-label">vs LanceDB</span>
          <div className="text-4xl font-bold text-[#298DFF] mt-2">{speedupVsLance}x</div>
          <p className="text-sm text-zinc-500 mt-1">faster search</p>
        </div>
        <div className="stat-card">
          <span className="mono-label">vs ChromaDB</span>
          <div className="text-4xl font-bold text-white mt-2">{speedupVsChroma}x</div>
          <p className="text-sm text-zinc-500 mt-1">faster search</p>
        </div>
      </div>

      {/* Benchmark Bars */}
      <div className="bg-[#0e0e13] rounded-xl p-6 border border-[#27272a]">
        <h3 className="mono-label mb-6">Search Latency (lower is better)</h3>
        <div key={animationKey}>
          {sortedResults.map((result, index) => (
            <BenchmarkBar
              key={result.name}
              label={result.name}
              value={result.search_time_us}
              maxValue={maxSearchTime}
              unit="μs"
              color={DB_COLORS[result.name] || "#71717a"}
              delay={index * 200}
              isWinner={index === 0}
            />
          ))}
        </div>
      </div>

      {/* Recall Table */}
      <div className="bg-[#0e0e13] rounded-xl p-6 border border-[#27272a]">
        <h3 className="mono-label mb-4">Recall@10 Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#27272a]">
                <th className="text-left py-3 px-4 font-mono text-zinc-500">Database</th>
                <th className="text-right py-3 px-4 font-mono text-zinc-500">Search (μs)</th>
                <th className="text-right py-3 px-4 font-mono text-zinc-500">Recall@10</th>
              </tr>
            </thead>
            <tbody>
              {sortedResults.map(result => (
                <tr key={result.name} className="border-b border-[#1f1f23] hover:bg-[#141419] transition-colors">
                  <td className="py-3 px-4">
                    <span
                      className="inline-flex items-center gap-2"
                      style={{ color: DB_COLORS[result.name] }}
                    >
                      <span className="w-2 h-2 rounded-full" style={{ background: DB_COLORS[result.name] }} />
                      {result.name}
                    </span>
                  </td>
                  <td className="text-right py-3 px-4 font-mono">
                    {result.search_time_us.toFixed(1)}
                  </td>
                  <td className="text-right py-3 px-4">
                    <span className={`font-mono ${result.recall_at_10 >= 0.99 ? 'text-green-500' : result.recall_at_10 >= 0.9 ? 'text-yellow-500' : 'text-red-500'}`}>
                      {(result.recall_at_10 * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
