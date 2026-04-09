import { RotateCcw, Home } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";
import { IconButton } from "@/components/ui/IconButton";

export function Toolbar() {
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const backToGalaxy = useGraphStore((s) => s.backToGalaxy);
  const loadOverview = useGraphStore((s) => s.loadOverview);
  const drillIntoCommunity = useGraphStore((s) => s.drillIntoCommunity);
  const activeCommunity = useGraphStore((s) => s.activeCommunity);
  const loading = useGraphStore((s) => s.loading);

  const handleRefresh = () => {
    if (zoomLevel === "cluster" && activeCommunity) {
      drillIntoCommunity(activeCommunity);
    } else {
      loadOverview();
    }
  };

  return (
    <div className="flex items-center gap-1">
      {/* Zoom level indicator */}
      <div className="flex items-center gap-1.5 px-2 py-1 bg-base rounded text-xs">
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            zoomLevel === "galaxy"
              ? "bg-accent"
              : zoomLevel === "cluster"
                ? "bg-success"
                : "bg-warning"
          }`}
        />
        <span className="text-text-secondary capitalize">{zoomLevel}</span>
      </div>

      <div className="w-px h-4 bg-border mx-1" />

      {/* Navigation */}
      {zoomLevel !== "galaxy" && (
        <IconButton
          icon={Home}
          tooltip="Back to Galaxy view"
          onClick={backToGalaxy}
        />
      )}

      <IconButton
        icon={RotateCcw}
        tooltip="Refresh"
        onClick={handleRefresh}
        disabled={loading}
        className={loading ? "[&_svg]:animate-spin" : ""}
      />

    </div>
  );
}
