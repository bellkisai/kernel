import { Network, ChevronLeft, Layers, Loader2 } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";
import { Panel } from "@/components/ui/Panel";
import { Badge } from "@/components/ui/Badge";
import { SIDEBAR_WIDTH } from "@/lib/layout";

export function CommunityPanel() {
  const clusters = useGraphStore((s) => s.clusters);
  const activeCommunity = useGraphStore((s) => s.activeCommunity);
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const drillIntoCommunity = useGraphStore((s) => s.drillIntoCommunity);
  const backToGalaxy = useGraphStore((s) => s.backToGalaxy);
  const loading = useGraphStore((s) => s.loading);

  const handleDrill = (label: string) => {
    if (loading) return;
    drillIntoCommunity(label);
  };

  return (
    <Panel variant="sidebar" width={SIDEBAR_WIDTH}>
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
        {zoomLevel !== "galaxy" ? (
          <button
            onClick={() => { if (!loading) backToGalaxy(); }}
            disabled={loading}
            className="p-1 hover:bg-overlay rounded cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas"
          >
            <ChevronLeft size={14} className="text-text-secondary" />
          </button>
        ) : (
          <Layers size={14} className="text-text-muted" />
        )}
        <h3 className="text-sm font-medium text-text-primary truncate">
          {zoomLevel === "galaxy"
            ? "Communities"
            : activeCommunity ?? "Cluster"}
        </h3>
        {loading && (
          <Loader2 size={12} className="text-accent animate-spin ml-auto shrink-0" />
        )}
      </div>

      {/* Cluster list */}
      <div className="flex-1 overflow-y-auto">
        {clusters.length === 0 ? (
          <div className="p-4 text-xs text-text-muted">
            No communities found. Store some memories first.
          </div>
        ) : (
          clusters.map((cluster) => {
            const isActive = activeCommunity === cluster.label;
            return (
              <button
                key={cluster.label}
                onClick={() => handleDrill(cluster.label)}
                disabled={loading}
                className={[
                  "w-full text-left px-4 py-3 transition-colors duration-micro cursor-pointer",
                  "border-b border-border-subtle",
                  "hover:bg-overlay/50",
                  "disabled:opacity-50 disabled:cursor-not-allowed",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas",
                  isActive
                    ? "bg-accent/10 border-l-2 !border-l-accent"
                    : "",
                ].join(" ")}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-primary truncate max-w-[160px]">
                    {cluster.label}
                  </span>
                  <span className="flex items-center gap-1 shrink-0 ml-2">
                    <Network size={10} className="text-text-muted" />
                    <Badge variant="count" count={cluster.member_count} />
                  </span>
                </div>
                {cluster.summary && (
                  <p className="mt-1 text-xs text-text-muted line-clamp-2">
                    {cluster.summary}
                  </p>
                )}
              </button>
            );
          })
        )}
      </div>

      {/* Back to galaxy (visible in cluster/neighborhood view) */}
      {zoomLevel !== "galaxy" && (
        <button
          onClick={() => { if (!loading) backToGalaxy(); }}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 border-t border-border text-xs text-accent hover:bg-accent/10 transition-colors duration-micro cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas"
        >
          <ChevronLeft size={12} />
          Back to Galaxy
        </button>
      )}

      {/* Footer stats */}
      <div className="px-4 py-2 border-t border-border text-xs text-text-disabled">
        {clusters.length} communities
      </div>
    </Panel>
  );
}
