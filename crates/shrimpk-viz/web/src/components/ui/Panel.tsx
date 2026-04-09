import type { ReactNode, CSSProperties } from "react";
import { SIDEBAR_WIDTH } from "@/lib/layout";

interface PanelProps {
  variant: "sidebar" | "overlay" | "legend";
  className?: string;
  children: ReactNode;
  style?: CSSProperties;
  width?: number;
}

const variantClasses = {
  sidebar: "bg-base border-r border-border flex flex-col overflow-hidden",
  overlay: "absolute bg-elevated rounded-lg shadow-xl z-overlays",
  legend: "absolute bg-base/80 backdrop-blur-sm rounded px-3 py-2 z-panels",
} as const;

export function Panel({ variant, className = "", children, style, width }: PanelProps) {
  const resolvedStyle: CSSProperties = { ...style };
  if (variant === "sidebar") {
    resolvedStyle.width = width ?? SIDEBAR_WIDTH;
  }

  return (
    <div
      className={`${variantClasses[variant]} ${className}`}
      style={resolvedStyle}
    >
      {children}
    </div>
  );
}
