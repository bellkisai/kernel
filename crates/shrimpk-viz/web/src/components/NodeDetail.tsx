import { X, Expand, Link } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";

const CATEGORY_COLORS: Record<string, string> = {
  Identity: "bg-blue-500/20 text-blue-400",
  Fact: "bg-green-500/20 text-green-400",
  Preference: "bg-purple-500/20 text-purple-400",
  ActiveProject: "bg-orange-500/20 text-orange-400",
  Conversation: "bg-slate-500/20 text-slate-400",
  Default: "bg-zinc-500/20 text-zinc-400",
};

function Bar({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-zinc-500">{label}</span>
      <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="w-8 text-right text-zinc-500">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function NodeDetail() {
  const selectedNode = useGraphStore((s) => s.selectedNode);
  const detail = useGraphStore((s) => s.selectedDetail);
  const selectNode = useGraphStore((s) => s.selectNode);
  const expandNode = useGraphStore((s) => s.expandNode);

  if (!selectedNode) return null;

  return (
    <div className="w-[350px] bg-zinc-900/95 border-l border-zinc-800 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <h3 className="text-sm font-medium text-zinc-200 truncate">
          Memory Detail
        </h3>
        <button
          onClick={() => selectNode(null)}
          className="p-1 hover:bg-zinc-800 rounded"
        >
          <X size={14} className="text-zinc-500" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {detail ? (
          <>
            {/* Category badge */}
            <span
              className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${CATEGORY_COLORS[detail.category] ?? CATEGORY_COLORS.Default}`}
            >
              {detail.category}
            </span>

            {/* Full content */}
            <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
              {detail.content}
            </p>

            {/* Labels */}
            {detail.labels.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {detail.labels.map((label) => (
                  <span
                    key={label}
                    className="px-2 py-0.5 bg-zinc-800 text-zinc-400 text-xs rounded"
                  >
                    {label}
                  </span>
                ))}
              </div>
            )}

            {/* Metrics */}
            <div className="space-y-2">
              <Bar value={detail.novelty_score} label="Novelty" />
              <div className="flex items-center gap-2 text-xs">
                <span className="w-20 text-zinc-500">Echo count</span>
                <span className="text-zinc-300">{detail.echo_count}</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-20 text-zinc-500">Source</span>
                <span className="text-zinc-300">{detail.source}</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="w-20 text-zinc-500">Created</span>
                <span className="text-zinc-300">
                  {new Date(detail.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-2">
              <button
                onClick={() => expandNode(selectedNode)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs rounded"
              >
                <Expand size={12} />
                Expand neighbors
              </button>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(detail.memory_id);
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded"
              >
                <Link size={12} />
                Copy ID
              </button>
            </div>
          </>
        ) : (
          <div className="text-sm text-zinc-500">Loading...</div>
        )}
      </div>
    </div>
  );
}
