"use client";

import { useEffect, useState } from "react";

export function ArchitectureDiagram() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep(prev => (prev + 1) % 4);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const steps = [
    { label: "Insert Vectors", description: "SIMD-optimized batch insertion" },
    { label: "MAP-Elites Evolution", description: "Quality-diversity optimization" },
    { label: "Index Selection", description: "Auto-select HNSW/Flat/IVF" },
    { label: "Optimized Search", description: "42μs search latency" },
  ];

  return (
    <div className="bg-[#0e0e13] rounded-xl p-8 border border-[#27272a]">
      <h3 className="mono-label mb-8 text-center">How EmergentDB Works</h3>

      {/* Pipeline visualization */}
      <div className="flex items-center justify-between mb-8">
        {steps.map((step, index) => (
          <div key={index} className="flex items-center">
            <div
              className={`relative flex flex-col items-center transition-all duration-500 ${
                activeStep === index ? 'scale-110' : 'opacity-60'
              }`}
            >
              {/* Circle */}
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center text-lg font-bold transition-all duration-500 ${
                  activeStep === index
                    ? 'bg-[#298DFF] text-white shadow-lg shadow-[#298DFF]/30'
                    : 'bg-[#141419] text-zinc-500 border border-[#27272a]'
                }`}
              >
                {index + 1}
              </div>
              {/* Label */}
              <span className="mt-3 text-sm font-medium text-center max-w-[100px]">
                {step.label}
              </span>
              {/* Description on active */}
              {activeStep === index && (
                <span className="absolute -bottom-10 text-xs text-[#298DFF] whitespace-nowrap">
                  {step.description}
                </span>
              )}
            </div>
            {/* Connector line */}
            {index < steps.length - 1 && (
              <div className="flex-1 mx-4 h-0.5 bg-[#27272a] relative min-w-[40px]">
                <div
                  className={`absolute inset-0 bg-[#298DFF] transition-transform duration-500 origin-left ${
                    activeStep > index ? 'scale-x-100' : 'scale-x-0'
                  }`}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Technical details */}
      <div className="grid grid-cols-3 gap-4 mt-12 pt-8 border-t border-[#27272a]">
        <div className="text-center">
          <div className="text-2xl font-bold text-[#298DFF]">6³</div>
          <div className="text-xs text-zinc-500 mt-1">MAP-Elites Grid Cells</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-white">99%+</div>
          <div className="text-xs text-zinc-500 mt-1">Recall Floor</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-white">5.6M</div>
          <div className="text-xs text-zinc-500 mt-1">Vectors/sec Insert</div>
        </div>
      </div>
    </div>
  );
}
