import { useGraphStore } from "@/stores/graphStore";
import { Panel } from "@/components/ui/Panel";
import { communityColor } from "@/lib/categoryColors";

export function CommunityLegend() {
  const communityMap = useGraphStore((s) => s.communityMap);
  const zoomLevel = useGraphStore((s) => s.zoomLevel);

  if (zoomLevel === "galaxy" || communityMap.size === 0) return null;

  const entries = Array.from(communityMap.entries())
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 10);

  return (
    <Panel variant="legend" className="bottom-4 left-4 flex flex-wrap gap-2 max-w-xs">
      {entries.map(([id, members], index) => (
        <span
          key={id}
          className="inline-flex items-center gap-1.5 text-text-secondary text-micro"
        >
          <span
            className="w-3 h-3 rounded-full shrink-0"
            style={{ backgroundColor: communityColor(id) }}
          />
          Cluster {index + 1} ({members.length})
        </span>
      ))}
    </Panel>
  );
}
