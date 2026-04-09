import { useState, useCallback, useRef, useEffect } from "react";
import { Search } from "lucide-react";
import { searchMemories, type EchoResult } from "../api/client";
import { useGraphStore } from "../stores/graphStore";

export function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<EchoResult[]>([]);
  const [open, setOpen] = useState(false);
  const expandNode = useGraphStore((s) => s.expandNode);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleCtrlK = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener("keydown", handleCtrlK);
    return () => document.removeEventListener("keydown", handleCtrlK);
  }, []);

  useEffect(() => {
    if (!open) return;
    const handleMouseDown = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, [open]);

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
    <div ref={wrapperRef} className="relative">
      <div className="flex items-center gap-2 bg-base border border-border rounded-lg px-3 py-1.5 focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-1 focus-within:ring-offset-canvas">
        <Search size={14} className="text-text-muted" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search memories... (Ctrl+K)"
          className="bg-transparent text-sm text-text-primary placeholder:text-text-disabled outline-none w-64"
        />
      </div>

      {/* Dropdown results */}
      {open && results.length > 0 && (
        <div className="absolute top-full left-0 mt-1 w-80 bg-elevated border border-border rounded-lg shadow-xl z-dropdowns max-h-80 overflow-y-auto">
          {results.map((r) => (
            <button
              key={r.memory_id}
              onClick={() => {
                expandNode(r.memory_id);
                setOpen(false);
                setQuery("");
              }}
              className="w-full text-left px-3 py-2 hover:bg-overlay/50 border-b border-border-subtle last:border-0 transition-colors duration-micro focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas"
            >
              <p className="text-xs text-text-secondary line-clamp-2">{r.content}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-micro text-text-disabled">
                  {(r.similarity * 100).toFixed(0)}% match
                </span>
                {r.labels.slice(0, 2).map((l) => (
                  <span
                    key={l}
                    className="text-micro px-1 bg-overlay text-text-muted rounded"
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
