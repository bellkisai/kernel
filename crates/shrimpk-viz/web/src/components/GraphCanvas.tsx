import { useEffect, useRef, useCallback } from "react";
import {
  SigmaContainer,
  useLoadGraph,
  useRegisterEvents,
  useSigma,
} from "@react-sigma/core";
import "@react-sigma/core/lib/react-sigma.min.css";
import forceAtlas2 from "graphology-layout-forceatlas2";
import { ZoomIn, ZoomOut } from "lucide-react";
import { useGraphStore } from "../stores/graphStore";
import { CATEGORY_HEX } from "@/lib/categoryColors";
import { useCameraTransition } from "@/hooks/useCameraTransition";
import { IconButton } from "@/components/ui/IconButton";
import { SizeLegend } from "./SizeLegend";
import { CommunityLegend } from "./CommunityLegend";

/** Named constants for WebGL reducer colors (hex required, not CSS vars) */
const DIMMED_NODE_COLOR = "#27272a";
const DIMMED_EDGE_COLOR = "#1c1c1e";

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
  const selectedNode = useGraphStore((s) => s.selectedNode);
  const { animateToNode } = useCameraTransition();

  // Keep refs to current zoom level and store graph so event handlers
  // always read the latest values without needing to re-register.
  const zoomRef = useRef(zoomLevel);
  zoomRef.current = zoomLevel;

  // Load graph into sigma whenever the store graph changes
  useEffect(() => {
    loadGraph(graph);
  }, [graph, loadGraph]);

  // Register click + hover events.
  // Read node attributes from sigma's own graph (the copy loadGraph imported)
  // so we never hit a stale-reference issue with the store graph.
  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => {
        const sigGraph = sigma.getGraph();
        const attrs = sigGraph.getNodeAttributes(node);
        if (zoomRef.current === "galaxy" && attrs.nodeType === "cluster") {
          drillIntoCommunity(node);
        } else if (attrs.nodeType === "memory") {
          selectNode(node);
        }
      },
      doubleClickNode: ({ node }) => {
        const sigGraph = sigma.getGraph();
        const attrs = sigGraph.getNodeAttributes(node);
        if (attrs.nodeType === "memory") {
          expandNode(node);
        }
      },
      enterNode: ({ node }) => setHoveredNode(node),
      leaveNode: () => setHoveredNode(null),
      clickStage: () => selectNode(null),
    });
  }, [registerEvents, sigma, selectNode, expandNode, drillIntoCommunity, setHoveredNode]);

  // Animate camera to center on the selected node.
  useEffect(() => {
    if (selectedNode) {
      animateToNode(selectedNode);
    }
  }, [selectedNode, animateToNode]);

  // Dim non-neighbor nodes on hover using sigma's nodeReducer, which
  // applies a visual override without mutating the graph data.
  useEffect(() => {
    sigma.setSetting("nodeReducer", (node, attrs) => {
      if (!hoveredNode) return attrs;
      const sigGraph = sigma.getGraph();
      const isNeighbor =
        node === hoveredNode || sigGraph.areNeighbors(node, hoveredNode);
      return isNeighbor ? attrs : { ...attrs, color: DIMMED_NODE_COLOR };
    });
    sigma.setSetting("edgeReducer", (edge, attrs) => {
      if (!hoveredNode) return attrs;
      const sigGraph = sigma.getGraph();
      const src = sigGraph.source(edge);
      const tgt = sigGraph.target(edge);
      const connected = src === hoveredNode || tgt === hoveredNode;
      return connected ? attrs : { ...attrs, color: DIMMED_EDGE_COLOR };
    });
  }, [hoveredNode, sigma]);

  return null;
}

/** Force-directed layout runner using ForceAtlas2. */
function LayoutRunner() {
  const sigma = useSigma();
  const graph = useGraphStore((s) => s.graph);
  const { animateToNodes } = useCameraTransition();
  const animFrame = useRef<number>();
  const iterRef = useRef(0);

  const stopLayout = useCallback(() => {
    if (animFrame.current) {
      cancelAnimationFrame(animFrame.current);
      animFrame.current = undefined;
    }
  }, []);

  useEffect(() => {
    stopLayout();
    iterRef.current = 0;

    const sigGraph = sigma.getGraph();
    if (!sigGraph || sigGraph.order < 2) return;

    const totalIterations = 80;

    const step = () => {
      if (iterRef.current >= totalIterations) {
        animFrame.current = undefined;
        // Fit camera to all nodes after layout stabilizes
        animateToNodes(sigGraph.nodes());
        return;
      }

      // Run a single ForceAtlas2 iteration directly on sigma's graph.
      // This mutates node x/y attributes in place.
      forceAtlas2.assign(sigGraph, {
        iterations: 1,
        settings: {
          gravity: 1,
          scalingRatio: 10,
          slowDown: 5,
          barnesHutOptimize: sigGraph.order > 50,
        },
      });

      sigma.refresh();
      iterRef.current++;
      animFrame.current = requestAnimationFrame(step);
    };

    // Kick off on the next frame so loadGraph has flushed into sigma.
    animFrame.current = requestAnimationFrame(step);

    return stopLayout;
  }, [graph, sigma, stopLayout, animateToNodes]);

  return null;
}

/** Zoom buttons rendered inside SigmaContainer so they can use useSigma(). */
function ZoomControls() {
  const sigma = useSigma();
  return (
    <div className="absolute top-2 right-2 flex flex-col gap-1 z-panels">
      <IconButton
        icon={ZoomIn}
        tooltip="Zoom in"
        onClick={() => {
          const camera = sigma.getCamera();
          camera.animate({ ratio: camera.ratio / 1.5 }, { duration: 250 });
        }}
      />
      <IconButton
        icon={ZoomOut}
        tooltip="Zoom out"
        onClick={() => {
          const camera = sigma.getCamera();
          camera.animate({ ratio: camera.ratio * 1.5 }, { duration: 250 });
        }}
      />
    </div>
  );
}

export function GraphCanvas() {
  return (
    <div className="flex-1 relative">
      <SigmaContainer
        className="sigma-container"
        settings={{
          defaultNodeColor: CATEGORY_HEX.Default,
          defaultEdgeColor: "#3f3f46",
          labelColor: { color: "#a1a1aa" },
          labelFont: "Inter, system-ui, sans-serif",
          labelSize: 11,
          labelRenderedSizeThreshold: 6,
          edgeLabelSize: 10,
          renderEdgeLabels: false,
          enableEdgeEvents: false,
          zoomDuration: 250,
          inertiaDuration: 300,
        }}
      >
        <GraphEvents />
        <LayoutRunner />
        <ZoomControls />
      </SigmaContainer>
      <SizeLegend />
      <CommunityLegend />
    </div>
  );
}
