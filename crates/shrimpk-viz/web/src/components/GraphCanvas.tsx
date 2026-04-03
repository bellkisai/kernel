import { useEffect, useRef } from "react";
import {
  SigmaContainer,
  useLoadGraph,
  useRegisterEvents,
  useSigma,
} from "@react-sigma/core";
import "@react-sigma/core/lib/react-sigma.min.css";
import { useGraphStore } from "../stores/graphStore";

/** Inner component that wires sigma events + loads graph data. */
function GraphEvents() {
  const sigma = useSigma();
  const loadGraph = useLoadGraph();
  const registerEvents = useRegisterEvents();
  const graph = useGraphStore((s) => s.graph);
  const zoomLevel = useGraphStore((s) => s.zoomLevel);
  const selectNode = useGraphStore((s) => s.selectNode);
  const expandNode = useGraphStore((s) => s.expandNode);
  const drillIntoCommunity = useGraphStore((s) => s.drillIntoCommunity);
  const setHoveredNode = useGraphStore((s) => s.setHoveredNode);
  const hoveredNode = useGraphStore((s) => s.hoveredNode);

  // Load graph whenever it changes
  useEffect(() => {
    loadGraph(graph);

    // Apply random initial positions if nodes don't have positions
    graph.forEachNode((node) => {
      if (!graph.getNodeAttribute(node, "x")) {
        graph.setNodeAttribute(node, "x", Math.random() * 1000 - 500);
        graph.setNodeAttribute(node, "y", Math.random() * 1000 - 500);
      }
    });
  }, [graph, loadGraph]);

  // Register click + hover events
  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => {
        const attrs = graph.getNodeAttributes(node);
        if (zoomLevel === "galaxy" && attrs.type === "cluster") {
          drillIntoCommunity(node);
        } else if (attrs.type === "memory") {
          selectNode(node);
        }
      },
      doubleClickNode: ({ node }) => {
        const attrs = graph.getNodeAttributes(node);
        if (attrs.type === "memory") {
          expandNode(node);
        }
      },
      enterNode: ({ node }) => setHoveredNode(node),
      leaveNode: () => setHoveredNode(null),
      clickStage: () => selectNode(null),
    });
  }, [
    registerEvents,
    graph,
    zoomLevel,
    selectNode,
    expandNode,
    drillIntoCommunity,
    setHoveredNode,
  ]);

  // Highlight hovered node's neighbors
  useEffect(() => {
    const s = sigma.getGraph();
    if (!s) return;

    s.forEachNode((node) => {
      if (!hoveredNode) {
        s.setNodeAttribute(node, "hidden", false);
        return;
      }
      const isNeighbor =
        node === hoveredNode || s.areNeighbors(node, hoveredNode);
      s.setNodeAttribute(node, "hidden", false);
      s.setNodeAttribute(
        node,
        "color",
        isNeighbor
          ? s.getNodeAttribute(node, "color")
          : "#27272a",
      );
    });
  }, [hoveredNode, sigma]);

  return null;
}

/** Force-directed layout runner. */
function LayoutRunner() {
  const sigma = useSigma();
  const graph = useGraphStore((s) => s.graph);
  const animFrame = useRef<number>();

  useEffect(() => {
    if (graph.order === 0) return;

    // Simple force-directed layout (ForceAtlas2 can be added via web worker)
    let iterations = 0;
    const maxIterations = 100;

    const step = () => {
      const g = sigma.getGraph();
      if (!g || iterations >= maxIterations) return;

      // Spring-electric model
      const positions: Record<string, { x: number; y: number }> = {};
      g.forEachNode((node) => {
        positions[node] = {
          x: g.getNodeAttribute(node, "x") || 0,
          y: g.getNodeAttribute(node, "y") || 0,
        };
      });

      // Repulsion between all nodes
      const nodes = Object.keys(positions);
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = positions[nodes[i]];
          const b = positions[nodes[j]];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
          const force = 500 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          a.x -= fx;
          a.y -= fy;
          b.x += fx;
          b.y += fy;
        }
      }

      // Attraction along edges
      g.forEachEdge((_edge, _attrs, source, target) => {
        const a = positions[source];
        const b = positions[target];
        if (!a || !b) return;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const force = dist * 0.01;
        const fx = (dx / Math.max(1, dist)) * force;
        const fy = (dy / Math.max(1, dist)) * force;
        a.x += fx;
        a.y += fy;
        b.x -= fx;
        b.y -= fy;
      });

      // Apply positions
      for (const [node, pos] of Object.entries(positions)) {
        g.setNodeAttribute(node, "x", pos.x);
        g.setNodeAttribute(node, "y", pos.y);
      }

      iterations++;
      if (iterations < maxIterations) {
        animFrame.current = requestAnimationFrame(step);
      }
    };

    animFrame.current = requestAnimationFrame(step);
    return () => {
      if (animFrame.current) cancelAnimationFrame(animFrame.current);
    };
  }, [graph, sigma]);

  return null;
}

export function GraphCanvas() {
  return (
    <div className="flex-1 relative">
      <SigmaContainer
        className="sigma-container"
        settings={{
          defaultNodeColor: "#71717a",
          defaultEdgeColor: "#3f3f46",
          labelColor: { color: "#a1a1aa" },
          labelFont: "Inter, system-ui, sans-serif",
          labelSize: 11,
          labelRenderedSizeThreshold: 6,
          edgeLabelSize: 10,
          renderEdgeLabels: false,
          enableEdgeEvents: false,
          zoomDuration: 200,
          inertiaDuration: 300,
        }}
      >
        <GraphEvents />
        <LayoutRunner />
      </SigmaContainer>
    </div>
  );
}
