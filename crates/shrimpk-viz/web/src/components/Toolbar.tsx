import { RotateCcw, ZoomIn, ZoomOut, Home } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";

export function Toolbar() {
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const backToGalaxy = useGraphStore((s) => s.backToGalaxy);
  const loading = useGraphStore((s) => s.loading);

  return (
    <div className="flex items-center gap-1">
      {/* Zoom level indicator */}
      <div className="flex items-center gap-1.5 px-2 py-1 bg-zinc-900 rounded text-xs">
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            zoomLevel === "galaxy"
              ? "bg-indigo-500"
              : zoomLevel === "cluster"
                ? "bg-green-500"
                : "bg-orange-500"
          }`}
        />
        <span className="text-zinc-400 capitalize">{zoomLevel}</span>
      </div>

      <div className="w-px h-4 bg-zinc-800 mx-1" />

      {/* Navigation */}
      {zoomLevel !== "galaxy" && (
        <button
          onClick={backToGalaxy}
          className="p-1.5 hover:bg-zinc-800 rounded text-zinc-500 hover:text-zinc-300"
          title="Back to Galaxy view"
        >
          <Home size={14} />
        </button>
      )}

      <button
        onClick={backToGalaxy}
        disabled={loading}
        className="p-1.5 hover:bg-zinc-800 rounded text-zinc-500 hover:text-zinc-300 disabled:opacity-30"
        title="Refresh"
      >
        <RotateCcw size={14} className={loading ? "animate-spin" : ""} />
      </button>

      {/* Placeholder for future zoom controls (sigma handles zoom via scroll) */}
      <button
        className="p-1.5 hover:bg-zinc-800 rounded text-zinc-500 hover:text-zinc-300"
        title="Zoom in (scroll up)"
      >
        <ZoomIn size={14} />
      </button>
      <button
        className="p-1.5 hover:bg-zinc-800 rounded text-zinc-500 hover:text-zinc-300"
        title="Zoom out (scroll down)"
      >
        <ZoomOut size={14} />
      </button>
    </div>
  );
}
