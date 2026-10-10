"""SLAM backend interface and a placeholder backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pocketslam.core.types import DepthMap, Frame, Pose


@runtime_checkable
class SlamBackend(Protocol):
    """Tracks the camera from RGB frames plus depth."""

    def track(self, frame: Frame, depth: DepthMap) -> Pose | None:
        """Process one frame.

        Args:
            frame: Input frame.
            depth: Depth for the same frame. Check ``depth.is_metric`` before
                using it as metres.

        Returns:
            Camera-to-world pose, or None if tracking is lost.
        """
        ...


class StaticSlam:
    """Placeholder that reports an identity pose for every frame.

    Lets the full pipeline run end to end before a real tracker is wired in.
    """

    def track(self, frame: Frame, depth: DepthMap) -> Pose | None:
        """Return the identity pose at the frame timestamp."""
        return Pose(timestamp_s=frame.timestamp_s)
