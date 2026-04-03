import { useState, useCallback } from "react";
import { Search } from "lucide-react";
import { searchMemories, type EchoResult } from "../api/client";
import { useGraphStore } from "../stores/graphStore";

export function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<EchoResult[]>([]);
  const [open, setOpen] = useState(false);
  const expandNode = useGraphStore((s) => s.expandNode);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) {
      setResults([]);
      setOpen(false);
      return;
    }
    try {
      const data = await searchMemories(query.trim(), 10);
      setResults(data.results);
      setOpen(true);
    } catch {
      setResults([]);
    }
  }, [query]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
    if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5">
        <Search size={14} className="text-zinc-500" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search memories..."
          className="bg-transparent text-sm text-zinc-200 placeholder:text-zinc-600 outline-none w-64"
        />
      </div>

      {/* Dropdown results */}
      {open && results.length > 0 && (
        <div className="absolute top-full left-0 mt-1 w-80 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto">
          {results.map((r) => (
            <button
              key={r.memory_id}
              onClick={() => {
                expandNode(r.memory_id);
                setOpen(false);
                setQuery("");
              }}
              className="w-full text-left px-3 py-2 hover:bg-zinc-800/50 border-b border-zinc-800/50 last:border-0"
            >
              <p className="text-xs text-zinc-300 line-clamp-2">{r.content}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-[10px] text-zinc-600">
                  {(r.similarity * 100).toFixed(0)}% match
                </span>
                {r.labels.slice(0, 2).map((l) => (
                  <span
                    key={l}
                    className="text-[10px] px-1 bg-zinc-800 text-zinc-500 rounded"
                  >
                    {l}
                  </span>
                ))}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
