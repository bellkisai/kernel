import { Network, ChevronLeft, Layers } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";

export function CommunityPanel() {
  const clusters = useGraphStore((s) => s.clusters);
  const activeCommunity = useGraphStore((s) => s.activeCommunity);
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const drillIntoCommunity = useGraphStore((s) => s.drillIntoCommunity);
  const backToGalaxy = useGraphStore((s) => s.backToGalaxy);

  return (
    <div className="w-[250px] bg-zinc-900/95 border-r border-zinc-800 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-800">
        {zoomLevel !== "galaxy" ? (
          <button
            onClick={backToGalaxy}
            className="p-1 hover:bg-zinc-800 rounded"
          >
            <ChevronLeft size={14} className="text-zinc-400" />
          </button>
        ) : (
          <Layers size={14} className="text-zinc-500" />
        )}
        <h3 className="text-sm font-medium text-zinc-200">
          {zoomLevel === "galaxy"
            ? "Communities"
            : activeCommunity ?? "Cluster"}
        </h3>
      </div>

      {/* Cluster list */}
      <div className="flex-1 overflow-y-auto">
        {clusters.length === 0 ? (
          <div className="p-4 text-xs text-zinc-500">
            No communities found. Store some memories first.
          </div>
        ) : (
          clusters.map((cluster) => (
            <button
              key={cluster.label}
              onClick={() => drillIntoCommunity(cluster.label)}
              className={`w-full text-left px-4 py-3 border-b border-zinc-800/50 hover:bg-zinc-800/50 transition-colors ${
                activeCommunity === cluster.label
                  ? "bg-indigo-500/10 border-l-2 border-l-indigo-500"
                  : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-200 truncate max-w-[160px]">
                  {cluster.label}
                </span>
                <span className="flex items-center gap-1 text-xs text-zinc-500">
                  <Network size={10} />
                  {cluster.member_count}
                </span>
              </div>
              {cluster.summary && (
                <p className="mt-1 text-xs text-zinc-500 line-clamp-2">
                  {cluster.summary}
                </p>
              )}
            </button>
          ))
        )}
      </div>

      {/* Footer stats */}
      <div className="px-4 py-2 border-t border-zinc-800 text-xs text-zinc-600">
        {clusters.length} communities
      </div>
    </div>
  );
}
