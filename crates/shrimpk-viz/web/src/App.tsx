import { useEffect, useState } from "react";
import { AppLayout } from "./layouts/AppLayout";
import { useGraphStore } from "./stores/graphStore";
import { checkHealth } from "./api/client";
import { LoadingState, ErrorState } from "@/components/ui/StateDisplay";

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
      <div className="h-screen flex items-center justify-center bg-canvas">
        <LoadingState message="Connecting to ShrimPK daemon..." />
      </div>
    );
  }

  if (!daemonOnline) {
    return (
      <div className="h-screen flex items-center justify-center bg-canvas">
        <ErrorState message="Daemon Offline" detail="ShrimPK daemon is not running on localhost:11435.">
          <code className="block text-xs text-text-disabled bg-base px-3 py-2 rounded">
            shrimpk-daemon
          </code>
          <p className="text-xs text-text-disabled">
            Retrying automatically...
          </p>
        </ErrorState>
      </div>
    );
  }

  return <AppLayout />;
}
