import { create } from "zustand";
import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import {
  fetchOverview,
  fetchMemoryGet,
  searchMemories,
  type GraphCluster,
  type GraphInterEdge,
  type MemoryDetail,
} from "../api/client";
import { getCategoryHex, CLUSTER_COLOR, ISOLATED_COLOR, communityColor } from "@/lib/categoryColors";

export type ZoomLevel = "galaxy" | "cluster" | "neighborhood";

/** Run Louvain and assign colors. Returns community→nodeIDs map. */
function assignCommunityColors(graph: Graph): Map<string, string[]> {
  const communityMap = new Map<string, string[]>();
  if (graph.order < 2 || graph.size === 0) {
    // No edges — mark all nodes with ISOLATED_COLOR
    graph.forEachNode((node) => {
      graph.setNodeAttribute(node, "color", ISOLATED_COLOR);
    });
    return communityMap;
  }
  louvain.assign(graph);
  graph.forEachNode((node, attrs) => {
    const community = attrs.community;
    if (community != null) {
      const key = String(community);
      graph.setNodeAttribute(node, "color", communityColor(community));
      const list = communityMap.get(key);
      if (list) list.push(node);
      else communityMap.set(key, [node]);
    } else {
      graph.setNodeAttribute(node, "color", ISOLATED_COLOR);
    }
  });
  return communityMap;
}

/** Log-scale node sizing: 10px (low importance) → 40px (high importance). */
function importanceToSize(importance: number, maxImportance: number): number {
  if (maxImportance <= 0) return 10;
  return 10 + 30 * (Math.log(importance + 1) / Math.log(maxImportance + 1));
}

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

  // Louvain community data
  communityMap: Map<string, string[]>;

  // UI state
  loading: boolean;
  error: string | null;
  detailError: string | null;
  daemonOnline: boolean;
  sidebarCollapsed: boolean;

  // Actions
  loadOverview: () => Promise<void>;
  drillIntoCommunity: (label: string) => Promise<void>;
  expandNode: (id: string) => Promise<void>;
  selectNode: (id: string | null) => Promise<void>;
  setHoveredNode: (id: string | null) => void;
  backToGalaxy: () => Promise<void>;
  setDaemonOnline: (online: boolean) => void;
  toggleSidebar: () => void;
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
  communityMap: new Map(),
  loading: false,
  error: null,
  detailError: null,
  daemonOnline: false,
  sidebarCollapsed: false,

  setDaemonOnline: (online) => set({ daemonOnline: online }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
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
        communityMap: new Map(),
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
      const cluster = get().clusters.find((c) => c.label === label);
      if (!cluster || cluster.top_members.length === 0) {
        set({ loading: false, error: `Cluster "${label}" not found or empty` });
        return;
      }

      // Use top member's content as semantic search query
      // (Hebbian graph endpoints deadlock, label string search returns 0)
      const query = cluster.top_members[0].content_preview;
      const data = await searchMemories(query, 100);

      if (!data.results || data.results.length === 0) {
        set({
          loading: false,
          error: `No memories found for cluster "${label}"`,
        });
        return;
      }

      const graph = new Graph();

      // Add member nodes with initial positions
      const count = data.results.length;
      const maxImportance = Math.max(...data.results.map((r) => r.final_score));
      for (let i = 0; i < count; i++) {
        const result = data.results[i];
        if (!graph.hasNode(result.memory_id)) {
          const angle = (2 * Math.PI * i) / count;
          const radius = 150 + Math.random() * 100;
          graph.addNode(result.memory_id, {
            label: result.content.slice(0, 40),
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            size: importanceToSize(result.final_score, maxImportance),
            color: getCategoryHex("Default"),
            nodeType: "memory",
            content: result.content,
            similarity: result.similarity,
          });
        }
      }

      // Create edges based on similarity proximity for community detection
      const nodeIds = graph.nodes();
      let edgeCount = 0;
      for (let i = 0; i < nodeIds.length && edgeCount < 200; i++) {
        for (let j = i + 1; j < nodeIds.length && edgeCount < 200; j++) {
          const a = graph.getNodeAttributes(nodeIds[i]);
          const b = graph.getNodeAttributes(nodeIds[j]);
          const weight = Math.min(a.similarity ?? 0, b.similarity ?? 0);
          if (weight > 0.3) {
            graph.addEdge(nodeIds[i], nodeIds[j], {
              weight,
              size: 0.5 + weight * 2,
              color: "#3f3f46",
            });
            edgeCount++;
          }
        }
      }

      // Assign community colors via Louvain (needs edges; falls back to isolated)
      const communityMap = assignCommunityColors(graph);

      set({
        graph,
        zoomLevel: "cluster",
        activeCommunity: label,
        communityMap,
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
      // Get center node details (memory_get works, neighbors deadlocks)
      const center = await fetchMemoryGet(id);

      // Find semantic neighbors via echo search
      const neighbors = await searchMemories(center.content.slice(0, 200), 20);
      const graph = new Graph();

      // Filter out the center node from results
      const neighborResults = neighbors.results.filter(r => r.memory_id !== id);

      // Compute maxImportance
      const maxImportance = Math.max(
        center.novelty_score ?? 0,
        ...neighborResults.map((n) => n.final_score),
      );

      // Add center node
      graph.addNode(center.memory_id, {
        label: center.content.slice(0, 40),
        x: 0,
        y: 0,
        size: importanceToSize(center.novelty_score ?? 0, maxImportance),
        color: getCategoryHex(center.category),
        nodeType: "memory",
        content: center.content,
        importance: center.novelty_score,
        category: center.category,
        isCenter: true,
      });

      // Add neighbors from echo results
      for (let i = 0; i < neighborResults.length; i++) {
        const result = neighborResults[i];
        if (!graph.hasNode(result.memory_id)) {
          const angle = (2 * Math.PI * i) / neighborResults.length;
          const radius = 100 + result.similarity * 200;
          graph.addNode(result.memory_id, {
            label: result.content.slice(0, 40),
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            size: importanceToSize(result.final_score, maxImportance),
            color: getCategoryHex("Default"),
            nodeType: "memory",
            content: result.content,
            weight: result.similarity,
          });
        }
        // Add edge from center to neighbor
        if (!graph.hasEdge(center.memory_id, result.memory_id)) {
          graph.addEdge(center.memory_id, result.memory_id, {
            weight: result.similarity,
            size: 0.5 + result.similarity * 2.5,
            color: result.similarity > 0.5 ? "#6366f1" : "#3f3f46",
          });
        }
      }

      // Assign community colors via Louvain
      const communityMap = assignCommunityColors(graph);

      set({
        graph,
        zoomLevel: "neighborhood",
        communityMap,
        selectedNode: id,
        selectedDetail: null,
        loading: false,
      });
      // Auto-select center node to show detail panel
      get().selectNode(id);
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
