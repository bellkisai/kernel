import { Layers, Search, Settings, type LucideIcon } from "lucide-react";

/** Panel width constants (px) */
export const SIDEBAR_WIDTH = 250;
export const SIDEBAR_COLLAPSED = 48;
export const DETAIL_PANEL_WIDTH = 350;
export const MIN_CANVAS_WIDTH = 400;

/** Icons shown in collapsed sidebar strip */
export const SIDEBAR_COLLAPSE_ICONS: { icon: LucideIcon; label: string; action: string }[] = [
  { icon: Layers, label: "Communities", action: "toggle-panel" },
  { icon: Search, label: "Search", action: "focus-search" },
  { icon: Settings, label: "Settings", action: "open-settings" },
];
