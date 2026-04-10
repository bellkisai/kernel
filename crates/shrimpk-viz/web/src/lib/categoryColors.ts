/**
 * Single source of truth for node category colors.
 * Non-graph design tokens live in globals.css @theme block.
 */

import { schemeTableau10 } from "d3-scale-chromatic";

/** Hex colors for Sigma WebGL rendering (CSS vars can't be read at paint time) */
export const CATEGORY_HEX: Record<string, string> = {
  Identity: "#3b82f6",
  Fact: "#22c55e",
  Preference: "#a855f7",
  ActiveProject: "#f97316",
  Conversation: "#64748b",
  Default: "#71717a",
};

/** Tailwind class strings for UI badge styling */
export const CATEGORY_BADGE: Record<string, string> = {
  Identity: "bg-blue-500/20 text-blue-400",
  Fact: "bg-green-500/20 text-green-400",
  Preference: "bg-purple-500/20 text-purple-400",
  ActiveProject: "bg-orange-500/20 text-orange-400",
  Conversation: "bg-slate-500/20 text-slate-400",
  Default: "bg-zinc-500/20 text-zinc-400",
};

/** Cluster super-node color */
export const CLUSTER_COLOR = "#6366f1";

/** Isolated/unconnected node color */
export const ISOLATED_COLOR = "#94a3b8";

/** Look up hex color for a category, falling back to Default */
export function getCategoryHex(category: string): string {
  return CATEGORY_HEX[category] ?? CATEGORY_HEX.Default;
}

/** Look up badge classes for a category, falling back to Default */
export function getCategoryBadge(category: string): string {
  return CATEGORY_BADGE[category] ?? CATEGORY_BADGE.Default;
}

// ---------------------------------------------------------------------------
// Community colors (Louvain detection → Tableau10 palette)
// ---------------------------------------------------------------------------

/** Deterministic hash for stable community→color mapping */
function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

/** Map a community ID to a Tableau10 color (deterministic) */
export function communityColor(communityId: string | number): string {
  return schemeTableau10[hashCode(String(communityId)) % 10];
}
