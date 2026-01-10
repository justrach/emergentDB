"use client";

import { useEffect, useState } from "react";

interface BenchmarkBarProps {
  label: string;
  value: number;
  maxValue: number;
  unit: string;
  color: string;
  delay?: number;
  isWinner?: boolean;
}

export function BenchmarkBar({
  label,
  value,
  maxValue,
  unit,
  color,
  delay = 0,
  isWinner = false,
}: BenchmarkBarProps) {
  const [width, setWidth] = useState(0);
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setWidth((value / maxValue) * 100);

      // Animate the counter
      const duration = 1500;
      const steps = 60;
      const increment = value / steps;
      let current = 0;

      const counterInterval = setInterval(() => {
        current += increment;
        if (current >= value) {
          setDisplayValue(value);
          clearInterval(counterInterval);
        } else {
          setDisplayValue(current);
        }
      }, duration / steps);

      return () => clearInterval(counterInterval);
    }, delay);

    return () => clearTimeout(timer);
  }, [value, maxValue, delay]);

  const formattedValue = unit === "μs"
    ? displayValue.toFixed(1)
    : displayValue.toFixed(0);

  return (
    <div className="mb-6">
      <div className="flex justify-between items-center mb-2">
        <span className="mono-label">{label}</span>
        <div className="flex items-center gap-2">
          <span
            className="font-mono text-lg font-semibold"
            style={{ color }}
          >
            {formattedValue}
            <span className="text-sm text-zinc-500 ml-1">{unit}</span>
          </span>
          {isWinner && (
            <span className="text-xs bg-[#298DFF]/20 text-[#298DFF] px-2 py-0.5 rounded-full font-medium">
              FASTEST
            </span>
          )}
        </div>
      </div>
      <div className="benchmark-bar">
        <div
          className="benchmark-bar-fill transition-all duration-[1500ms] ease-out"
          style={{
            width: `${width}%`,
            background: `linear-gradient(90deg, ${color} 0%, ${color}80 100%)`,
            boxShadow: isWinner ? `0 0 20px ${color}40` : 'none',
          }}
        />
      </div>
    </div>
  );
}
