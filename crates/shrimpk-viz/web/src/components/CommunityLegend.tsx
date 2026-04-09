import { useGraphStore } from "@/stores/graphStore";
import { Panel } from "@/components/ui/Panel";
import { communityColor } from "@/lib/categoryColors";

export function CommunityLegend() {
  const communityMap = useGraphStore((s) => s.communityMap);
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const clusters = useGraphStore((s) => s.clusters);
  const visible = zoomLevel !== "galaxy" && communityMap.size > 0;

  const entries = Array.from(communityMap.entries())
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 10);

  /** Try to match a community's members to a cluster with a summary. */
  function labelForCommunity(members: string[], index: number): string {
    for (const cluster of clusters) {
      if (!cluster.summary) continue;
      const memberSet = new Set(members);
      const hasOverlap = cluster.top_members.some((m) => memberSet.has(m.id));
      if (hasOverlap) {
        return cluster.summary.length > 20
          ? cluster.summary.slice(0, 20) + "\u2026"
          : cluster.summary;
      }
    }
    return `Group ${String.fromCharCode(65 + index)}`;
  }

  return (
    <Panel
      variant="legend"
      aria-hidden={!visible}
      className={`bottom-4 left-4 flex flex-wrap gap-2 max-w-xs transition-opacity duration-panel ease-out motion-reduce:transition-none ${visible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
    >
      {entries.map(([id, members], index) => (
        <span
          key={id}
          className="inline-flex items-center gap-1.5 text-text-secondary text-micro"
        >
          <span
            className="w-3 h-3 rounded-full shrink-0"
            style={{ backgroundColor: communityColor(id) }}
          />
          {labelForCommunity(members, index)} ({members.length})
        </span>
      ))}
    </Panel>
  );
}
