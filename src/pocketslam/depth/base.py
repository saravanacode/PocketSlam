"""Depth estimator interface and a placeholder backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from pocketslam.core.types import DepthMap, Frame


@runtime_checkable
class DepthEstimator(Protocol):
    """Anything that turns one RGB frame into a depth map."""

    def estimate(self, frame: Frame) -> DepthMap:
        """Estimate depth for ``frame``.

        Args:
            frame: Input frame.

        Returns:
            Depth with the same height and width as ``frame.image``.
        """
        ...


class ConstantDepthEstimator:
    """Placeholder that returns constant relative depth.

    Lets the full pipeline run end to end before a real model is wired in.
    Replace it with a real backend (for example Depth Anything or UniDepth).
    """

    def estimate(self, frame: Frame) -> DepthMap:
        """Return relative depth of 1.0 everywhere."""
        return DepthMap(
            depth=np.ones((frame.height, frame.width), dtype=np.float32),
            is_metric=False,
            timestamp_s=frame.timestamp_s,
        )
