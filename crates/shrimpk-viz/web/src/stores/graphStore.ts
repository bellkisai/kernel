import { create } from "zustand";
import Graph from "graphology";
import {
  fetchOverview,
  fetchNeighbors,
  fetchMemoryGet,
  fetchRelated,
  type GraphCluster,
  type GraphInterEdge,
  type MemoryDetail,
} from "../api/client";

// ---------------------------------------------------------------------------
// Category → color mapping (matches TUI palette)
// ---------------------------------------------------------------------------

const CATEGORY_COLORS: Record<string, string> = {
  Identity: "#3b82f6",
  Fact: "#22c55e",
  Preference: "#a855f7",
  ActiveProject: "#f97316",
  Conversation: "#64748b",
  Default: "#71717a",
};

const CLUSTER_COLOR = "#6366f1";

export type ZoomLevel = "galaxy" | "cluster" | "neighborhood";

interface GraphState {
  // Graph data
  graph: Graph;
  zoomLevel: ZoomLevel;

  // Selection
  selectedNode: string | null;
  selectedDetail: MemoryDetail | null;
  hoveredNode: string | null;

  // Community data (Galaxy view)
  clusters: GraphCluster[];
  interEdges: GraphInterEdge[];
  activeCommunity: string | null;

  // UI state
  loading: boolean;
  error: string | null;
  detailError: string | null;
  daemonOnline: boolean;

  // Actions
  loadOverview: () => Promise<void>;
  drillIntoCommunity: (label: string) => Promise<void>;
  expandNode: (id: string) => Promise<void>;
  selectNode: (id: string | null) => Promise<void>;
  setHoveredNode: (id: string | null) => void;
  backToGalaxy: () => Promise<void>;
  setDaemonOnline: (online: boolean) => void;
}

export const useGraphStore = create<GraphState>((set, get) => ({
  graph: new Graph(),
  zoomLevel: "galaxy",
  selectedNode: null,
  selectedDetail: null,
  hoveredNode: null,
  clusters: [],
  interEdges: [],
  activeCommunity: null,
  loading: false,
  error: null,
  detailError: null,
  daemonOnline: false,

  setDaemonOnline: (online) => set({ daemonOnline: online }),
  setHoveredNode: (id) => set({ hoveredNode: id }),

  loadOverview: async () => {
    set({ loading: true, error: null });
    try {
      const data = await fetchOverview();
      const graph = new Graph();

      // Add cluster super-nodes with initial positions
      for (let i = 0; i < data.clusters.length; i++) {
        const cluster = data.clusters[i];
        const angle = (2 * Math.PI * i) / data.clusters.length;
        const radius = 200;
        graph.addNode(cluster.label, {
          label: cluster.label,
          x: Math.cos(angle) * radius + (Math.random() - 0.5) * 40,
          y: Math.sin(angle) * radius + (Math.random() - 0.5) * 40,
          size: Math.max(8, Math.min(30, cluster.member_count / 2)),
          color: CLUSTER_COLOR,
          nodeType: "cluster",
          memberCount: cluster.member_count,
          summary: cluster.summary,
        });
      }

      // Add inter-cluster edges
      for (const edge of data.inter_edges) {
        if (graph.hasNode(edge.source_label) && graph.hasNode(edge.target_label)) {
          graph.addEdge(edge.source_label, edge.target_label, {
            weight: edge.shared_count,
            size: Math.max(1, Math.min(5, edge.shared_count / 3)),
            color: "#3f3f46",
          });
        }
      }

      set({
        graph,
        clusters: data.clusters,
        interEdges: data.inter_edges,
        zoomLevel: "galaxy",
        activeCommunity: null,
        selectedNode: null,
        selectedDetail: null,
        loading: false,
      });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  drillIntoCommunity: async (label) => {
    // Prevent duplicate concurrent calls
    if (get().loading) return;
    set({ loading: true, error: null });
    try {
      // Find a member ID to anchor the query
      const cluster = get().clusters.find((c) => c.label === label);
      if (!cluster) {
        set({ loading: false, error: `Cluster "${label}" not found` });
        return;
      }
      if (cluster.top_members.length === 0) {
        set({ loading: false, error: `Cluster "${label}" has no members` });
        return;
      }

      const anchorId = cluster.top_members[0].id;
      const data = await fetchRelated(anchorId, label, 100);

      if (!data.results || data.results.length === 0) {
        set({
          loading: false,
          error: `No related memories found for cluster "${label}"`,
        });
        return;
      }

      const graph = new Graph();

      // Add member nodes with initial positions
      const count = data.results.length;
      for (let i = 0; i < count; i++) {
        const result = data.results[i];
        if (!graph.hasNode(result.memory_id)) {
          const angle = (2 * Math.PI * i) / count;
          const radius = 150 + Math.random() * 100;
          graph.addNode(result.memory_id, {
            label: result.content.slice(0, 40),
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            size: 6 + result.final_score * 8,
            color: "#71717a",
            nodeType: "memory",
            content: result.content,
            similarity: result.similarity,
          });
        }
      }

      set({
        graph,
        zoomLevel: "cluster",
        activeCommunity: label,
        selectedNode: null,
        selectedDetail: null,
        loading: false,
      });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  expandNode: async (id) => {
    set({ loading: true, error: null });
    try {
      const data = await fetchNeighbors(id);
      const graph = new Graph();

      // Add center node
      graph.addNode(data.node.id, {
        label: data.node.content_preview.slice(0, 40),
        x: 0,
        y: 0,
        size: 14,
        color: CATEGORY_COLORS[data.node.category] ?? "#71717a",
        nodeType: "memory",
        content: data.node.content_preview,
        importance: data.node.importance,
        category: data.node.category,
        isCenter: true,
      });

      // Add neighbors
      for (let i = 0; i < data.neighbors.length; i++) {
        const neighbor = data.neighbors[i];
        if (!graph.hasNode(neighbor.id)) {
          const angle = (2 * Math.PI * i) / data.neighbors.length;
          const radius = 100 + neighbor.weight * 200;
          graph.addNode(neighbor.id, {
            label: neighbor.content_preview.slice(0, 40),
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            size: 4 + neighbor.weight * 10,
            color: "#71717a",
            nodeType: "memory",
            content: neighbor.content_preview,
            weight: neighbor.weight,
          });
        }
        if (!graph.hasEdge(data.node.id, neighbor.id)) {
          graph.addEdge(data.node.id, neighbor.id, {
            weight: neighbor.weight,
            size: 0.5 + neighbor.weight * 2.5,
            color: neighbor.relationship ? "#6366f1" : "#3f3f46",
            relationship: neighbor.relationship,
          });
        }
      }

      set({
        graph,
        zoomLevel: "neighborhood",
        selectedNode: null,
        selectedDetail: null,
        loading: false,
      });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  selectNode: async (id) => {
    if (!id) {
      set({ selectedNode: null, selectedDetail: null, detailError: null });
      return;
    }
    set({ selectedNode: id, selectedDetail: null, detailError: null });
    try {
      const detail = await fetchMemoryGet(id);
      // Guard against stale response — only apply if this node is still selected
      if (get().selectedNode === id) {
        set({ selectedDetail: detail });
      }
    } catch (e) {
      if (get().selectedNode === id) {
        set({ detailError: (e as Error).message });
      }
    }
  },

  backToGalaxy: async () => {
    await get().loadOverview();
  },
}));
