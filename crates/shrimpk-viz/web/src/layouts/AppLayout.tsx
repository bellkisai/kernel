import { SearchBar } from "../components/SearchBar";
import { Toolbar } from "../components/Toolbar";
import { CommunityPanel } from "../components/CommunityPanel";
import { GraphCanvas } from "../components/GraphCanvas";
import { NodeDetail } from "../components/NodeDetail";
import { useGraphStore } from "../stores/graphStore";

export function AppLayout() {
  const error = useGraphStore((s) => s.error);

  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-zinc-100">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 bg-zinc-900/80">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold tracking-tight text-indigo-400">
            ShrimPK
          </span>
          <span className="text-xs text-zinc-600">Knowledge Graph</span>
          <SearchBar />
        </div>
        <Toolbar />
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-950/50 border-b border-red-900/50 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        <CommunityPanel />
        <GraphCanvas />
        <NodeDetail />
      </div>
    </div>
  );
}
