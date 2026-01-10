"use client";

import { ReactNode } from "react";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  highlight?: boolean;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon,
  highlight = false,
  trend,
  trendValue,
}: StatCardProps) {
  const trendColors = {
    up: "text-green-500",
    down: "text-red-500",
    neutral: "text-zinc-500",
  };

  const trendIcons = {
    up: "↑",
    down: "↓",
    neutral: "→",
  };

  return (
    <div className={`stat-card ${highlight ? 'glow-accent border-[#298DFF]' : ''}`}>
      <div className="flex items-start justify-between mb-3">
        <span className="mono-label">{title}</span>
        {icon && <div className="text-zinc-500">{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <span className={`text-3xl font-bold ${highlight ? 'text-[#298DFF]' : 'text-white'}`}>
          {value}
        </span>
        {trend && trendValue && (
          <span className={`text-sm ${trendColors[trend]}`}>
            {trendIcons[trend]} {trendValue}
          </span>
        )}
      </div>
      {subtitle && (
        <p className="text-sm text-zinc-500 mt-2">{subtitle}</p>
      )}
    </div>
  );
}
