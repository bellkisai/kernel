import type { ReactNode } from "react";
import { getCategoryBadge } from "@/lib/categoryColors";

type BadgeProps =
  | { variant: "label"; children: ReactNode; className?: string }
  | { variant: "category"; category: string; children: ReactNode; className?: string }
  | { variant: "count"; count: number; className?: string };

export function Badge(props: BadgeProps) {
  const base = "inline-flex items-center justify-center rounded text-caption font-medium";

  if (props.variant === "label") {
    return (
      <span className={`${base} px-2 py-0.5 bg-overlay text-text-secondary ${props.className ?? ""}`}>
        {props.children}
      </span>
    );
  }

  if (props.variant === "category") {
    return (
      <span className={`${base} px-2 py-0.5 ${getCategoryBadge(props.category)} ${props.className ?? ""}`}>
        {props.children}
      </span>
    );
  }

  // count variant
  return (
    <span title={`${props.count} members`} className={`${base} min-w-[20px] h-5 px-1.5 bg-overlay text-text-secondary ${props.className ?? ""}`}>
      {props.count}
    </span>
  );
}
