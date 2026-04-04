import { useEffect, useState } from "react";
import { AppLayout } from "./layouts/AppLayout";
import { useGraphStore } from "./stores/graphStore";
import { checkHealth } from "./api/client";

export function App() {
  const loadOverview = useGraphStore((s) => s.loadOverview);
  const setDaemonOnline = useGraphStore((s) => s.setDaemonOnline);
  const daemonOnline = useGraphStore((s) => s.daemonOnline);
  const [checking, setChecking] = useState(true);

  // Check daemon health on mount, retry every 3s if offline
  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval>;

    const check = async () => {
      const online = await checkHealth();
      if (cancelled) return;
      setDaemonOnline(online);
      setChecking(false);
      if (online) {
        clearInterval(interval);
        await loadOverview();
      }
    };

    check();
    interval = setInterval(check, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [loadOverview, setDaemonOnline]);

  if (checking) {
    return (
      <div className="h-screen flex items-center justify-center bg-zinc-950">
        <div className="text-center space-y-3">
          <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-zinc-500">
            Connecting to ShrimPK daemon...
          </p>
        </div>
      </div>
    );
  }

  if (!daemonOnline) {
    return (
      <div className="h-screen flex items-center justify-center bg-zinc-950">
        <div className="text-center space-y-3 max-w-sm">
          <div className="w-10 h-10 bg-red-950 rounded-full flex items-center justify-center mx-auto">
            <span className="text-red-400 text-lg">!</span>
          </div>
          <h2 className="text-lg font-medium text-zinc-200">
            Daemon Offline
          </h2>
          <p className="text-sm text-zinc-500">
            ShrimPK daemon is not running on localhost:11435.
          </p>
          <code className="block text-xs text-zinc-600 bg-zinc-900 px-3 py-2 rounded">
            shrimpk-daemon
          </code>
          <p className="text-xs text-zinc-600">
            Retrying automatically...
          </p>
        </div>
      </div>
    );
  }

  return <AppLayout />;
}
