import { useSigma } from "@react-sigma/core";
import { useCallback } from "react";
import { getCameraStateToFitViewportToNodes } from "@sigma/utils";

const DURATION = 250;
const EASING = "quadraticOut" as const;

export function useCameraTransition() {
  const sigma = useSigma();

  const animateToNodes = useCallback(
    (nodeIds: string[]) => {
      const graph = sigma.getGraph();
      const validIds = nodeIds.filter((id) => graph.hasNode(id));

      if (validIds.length === 0) return;

      if (validIds.length === 1) {
        const attrs = graph.getNodeAttributes(validIds[0]);
        sigma.getCamera().animate(
          { x: attrs.x, y: attrs.y, ratio: 0.5 },
          { duration: DURATION, easing: EASING },
        );
        return;
      }

      // Multiple nodes — use @sigma/utils to compute the target camera state,
      // then animate with our own duration/easing for consistency.
      const targetState = getCameraStateToFitViewportToNodes(
        sigma,
        validIds,
      );
      sigma.getCamera().animate(targetState, {
        duration: DURATION,
        easing: EASING,
      });
    },
    [sigma],
  );

  const animateToNode = useCallback(
    (nodeId: string) => animateToNodes([nodeId]),
    [animateToNodes],
  );

  return { animateToNodes, animateToNode };
}
