import { useEffect } from "react";
import { X, Expand, Link } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { IconButton } from "@/components/ui/IconButton";
import { ErrorState, LoadingState } from "@/components/ui/StateDisplay";
import { DETAIL_PANEL_WIDTH } from "@/lib/layout";

function Bar({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-text-muted">{label}</span>
      <div className="flex-1 h-1.5 bg-overlay rounded-full overflow-hidden">
        <div
          className="h-full bg-accent rounded-full transition-all duration-panel"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-8 text-right text-text-muted">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function NodeDetail() {
  const selectedNode = useGraphStore((s) => s.selectedNode);
  const detail = useGraphStore((s) => s.selectedDetail);
  const detailError = useGraphStore((s) => s.detailError);
  const selectNode = useGraphStore((s) => s.selectNode);
  const expandNode = useGraphStore((s) => s.expandNode);

  useEffect(() => {
    if (!selectedNode) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") selectNode(null);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedNode, selectNode]);

  if (!selectedNode) return null;

  return (
    <div
      className="shrink-0 bg-base/95 border-l border-border flex flex-col overflow-hidden animate-slide-in"
      style={{ width: DETAIL_PANEL_WIDTH }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-primary truncate">
          Memory Detail
        </h3>
        <IconButton icon={X} tooltip="Close" size="sm" onClick={() => selectNode(null)} />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {detailError ? (
          <ErrorState
            message="Failed to load memory"
            detail={detailError}
            onRetry={() => selectNode(selectedNode)}
          />
        ) : detail ? (
          <>
            {/* Category badge */}
            <Badge variant="category" category={detail.category}>
              {detail.category}
            </Badge>

            {/* Full content */}
            <p className="text-sm text-text-secondary leading-relaxed whitespace-pre-wrap">
              {detail.content}
            </p>

            {/* Labels */}
            {detail.labels && detail.labels.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {detail.labels.map((label) => (
                  <Badge key={label} variant="label">
                    {label}
                  </Badge>
                ))}
              </div>
            )}

            {/* Metrics */}
            <div className="space-y-2 pt-1">
              <Bar value={detail.novelty_score ?? 0} label="Novelty" />
              <div className="flex items-center gap-2 text-xs">
                <span className="w-20 text-text-muted">Echo count</span>
                <span className="text-text-secondary">{detail.echo_count ?? 0}</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-20 text-text-muted">Source</span>
                <span className="text-text-secondary">{detail.source ?? "unknown"}</span>
              </div>
              {detail.modality && (
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-20 text-text-muted">Modality</span>
                  <span className="text-text-secondary">{detail.modality}</span>
                </div>
              )}
              <div className="flex items-center gap-2 text-xs">
                <span className="w-20 text-text-muted">Created</span>
                <span className="text-text-secondary">
                  {detail.created_at
                    ? new Date(detail.created_at).toLocaleDateString()
                    : "unknown"}
                </span>
              </div>
            </div>

            {/* ID (truncated) */}
            <div className="text-xs text-text-disabled font-mono truncate pt-1">
              {detail.memory_id}
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-2">
              <Button
                variant="primary"
                size="sm"
                onClick={() => expandNode(selectedNode)}
              >
                <Expand size={12} />
                Expand neighbors
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  navigator.clipboard.writeText(detail.memory_id);
                }}
              >
                <Link size={12} />
                Copy ID
              </Button>
            </div>
          </>
        ) : (
          <LoadingState message="Loading memory..." />
        )}
      </div>
    </div>
  );
}
