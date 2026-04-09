import { useGraphStore } from "@/stores/graphStore";
import { Panel } from "@/components/ui/Panel";

const CIRCLES = [
  { size: 10, label: "Low" },
  { size: 25, label: "Med" },
  { size: 40, label: "High" },
] as const;

export function SizeLegend() {
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const visible = zoomLevel !== "galaxy";

  return (
    <Panel
      variant="legend"
      aria-hidden={!visible}
      className={`bottom-4 right-4 flex items-end gap-3 border border-border transition-opacity duration-panel ease-out motion-reduce:transition-none ${visible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
    >
      {CIRCLES.map(({ size, label }) => (
        <div key={label} className="flex flex-col items-center gap-1">
          <div
            className="rounded-full bg-text-muted/40"
            style={{ width: size, height: size }}
          />
          <span className="text-text-muted text-micro leading-none">
            {label}
          </span>
        </div>
      ))}
    </Panel>
  );
}
