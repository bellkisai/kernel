const BASE = "http://localhost:11435";

// ---------------------------------------------------------------------------
// Types matching daemon JSON responses
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  content_preview: string;
  labels: string[];
  importance: number;
  category: string;
  novelty: number;
}

export interface GraphNeighbor {
  id: string;
  content_preview: string;
  labels: string[];
  weight: number;
  relationship: string | null;
  cosine_similarity: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  relationship: string | null;
}

export interface GraphCluster {
  label: string;
  member_count: number;
  summary: string | null;
  top_members: { id: string; content_preview: string }[];
}

export interface GraphInterEdge {
  source_label: string;
  target_label: string;
  shared_count: number;
}

export interface NeighborsResponse {
  node: GraphNode;
  neighbors: GraphNeighbor[];
  count: number;
}

export interface SubgraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
}

export interface OverviewResponse {
  clusters: GraphCluster[];
  inter_edges: GraphInterEdge[];
  cluster_count: number;
  edge_count: number;
}

export interface MemoryDetail {
  memory_id: string;
  content: string;
  source: string;
  modality: string;
  labels: string[];
  echo_count: number;
  created_at: string;
  category: string;
  sensitivity: string;
  novelty_score: number;
}

export interface EchoResult {
  rank: number;
  memory_id: string;
  content: string;
  similarity: number;
  final_score: number;
  source: string;
  labels: string[];
}

export interface Stats {
  total_memories: number;
  text_count: number;
  vision_count: number;
  speech_count: number;
  avg_echo_latency_ms: number;
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Galaxy view — label clusters + inter-cluster edges. */
export function fetchOverview(
  minMembers = 3,
  maxClusters = 30,
): Promise<OverviewResponse> {
  return post("/api/graph/overview", {
    min_members: minMembers,
    max_clusters: maxClusters,
  });
}

/** Neighborhood — Hebbian neighbors for one memory. */
export function fetchNeighbors(
  memoryId: string,
  minWeight = 0.05,
  maxResults = 50,
): Promise<NeighborsResponse> {
  return post("/api/graph/neighbors", {
    memory_id: memoryId,
    min_weight: minWeight,
    max_results: maxResults,
  });
}

/** Subgraph — batch node + edge fetch. */
export function fetchSubgraph(
  memoryIds: string[],
  includeNeighbors = true,
  minWeight = 0.05,
): Promise<SubgraphResponse> {
  return post("/api/graph/subgraph", {
    memory_ids: memoryIds,
    include_neighbors: includeNeighbors,
    min_weight: minWeight,
  });
}

/** Full memory content for detail panel. */
export function fetchMemoryGet(memoryId: string): Promise<MemoryDetail> {
  return post("/api/memory_get", { memory_id: memoryId });
}

/** Entity/text search via echo. */
export function searchMemories(
  query: string,
  maxResults = 20,
): Promise<{ results: EchoResult[]; count: number; elapsed_ms: number }> {
  return post("/api/echo", { query, max_results: maxResults });
}

/** Related memories via shared labels. */
export function fetchRelated(
  memoryId: string,
  label?: string,
  maxResults = 20,
): Promise<{ results: EchoResult[]; count: number }> {
  return post("/api/memory_related", {
    memory_id: memoryId,
    label,
    max_results: maxResults,
  });
}

/** System stats. */
export function fetchStats(): Promise<Stats> {
  return get("/api/stats");
}

/** Health check — returns true if daemon is reachable. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}
