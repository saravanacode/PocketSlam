"""Depth Anything 3 backends: multi-view depth + poses, and metric monocular depth.

Needs the optional ``recon`` extra: ``uv sync --extra recon``.

The ``depth-anything-3`` package imports its export helpers (COLMAP, video,
Gaussian splats) when ``depth_anything_3.api`` loads. We never export through
it, and one of those helpers pulls in ``pycolmap``, whose bundled OpenMP
runtime aborts the process next to torch's. ``_load_api`` therefore registers
a stand-in for that module before importing the API.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from pocketslam.core.types import MultiViewGeometry

logger = logging.getLogger(__name__)

# DA3 metric checkpoints output "canonical" depth: metres = output * focal / 300.
METRIC_CANONICAL_FOCAL = 300.0


class _Da3Model(Protocol):
    def inference(self, image: list[NDArray[np.uint8]], **kwargs: Any) -> Any: ...


def _load_api() -> Any:
    if "depth_anything_3.utils.export" not in sys.modules:
        stub = types.ModuleType("depth_anything_3.utils.export")

        def export(*_args: Any, **_kwargs: Any) -> None:
            msg = "DA3 export helpers are disabled in PocketSlam; use pocketslam.recon"
            raise NotImplementedError(msg)

        stub.export = export  # type: ignore[attr-defined]
        sys.modules["depth_anything_3.utils.export"] = stub
    try:
        from depth_anything_3.api import DepthAnything3  # noqa: PLC0415 - optional extra
    except ImportError as exc:
        msg = "Depth Anything 3 is not installed. Run: uv sync --extra recon"
        raise RuntimeError(msg) from exc
    return DepthAnything3


def load_model(model_id: str, revision: str, device: torch.device) -> _Da3Model:
    """Download (once) and load a DA3 checkpoint pinned to ``revision``."""
    api = _load_api()
    logger.info("loading depth model", extra={"model": model_id, "revision": revision})
    model = api.from_pretrained(model_id, revision=revision)
    return model.to(device).eval()  # type: ignore[no-any-return]


class DA3MultiView:
    """Joint depth, camera poses and intrinsics for a set of frames (any-view model)."""

    def __init__(self, model: _Da3Model, process_res: int = 504) -> None:
        self._model = model
        self._process_res = process_res

    def estimate_views(self, images: Sequence[NDArray[np.uint8]]) -> MultiViewGeometry:
        """Run the model on ``(H, W, 3)`` uint8 RGB frames.

        Returns:
            Geometry at the model's processing resolution, up to an unknown scale.
        """
        if len(images) < 2:  # noqa: PLR2004 - poses need two views
            msg = "need at least two frames to estimate camera poses"
            raise ValueError(msg)
        with torch.inference_mode():
            pred = self._model.inference(list(images), process_res=self._process_res)
        n = len(images)
        w2c = np.tile(np.eye(4), (n, 1, 1))
        w2c[:, :3, :4] = np.asarray(pred.extrinsics, dtype=np.float64)
        return MultiViewGeometry(
            depth=np.asarray(pred.depth, dtype=np.float32),
            confidence=np.asarray(pred.conf, dtype=np.float32),
            intrinsics=np.asarray(pred.intrinsics, dtype=np.float64),
            world_to_camera=w2c,
            images=np.asarray(pred.processed_images, dtype=np.uint8),
            is_metric=False,
        )


class DA3Metric:
    """Metric monocular depth (DA3METRIC checkpoints), one frame at a time."""

    def __init__(self, model: _Da3Model, process_res: int = 504) -> None:
        self._model = model
        self._process_res = process_res

    def metric_depth(
        self, images: Sequence[NDArray[np.uint8]], focal_px: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        """Depth in metres for each frame.

        Args:
            images: ``(H, W, 3)`` uint8 RGB frames.
            focal_px: ``(N,)`` focal length in pixels at the processing
                resolution, for example the mean of fx and fy from ``DA3MultiView``.

        Returns:
            ``(N, H, W)`` float32 depth in metres.
        """
        out = []
        with torch.inference_mode():
            for image in images:
                pred = self._model.inference([image], process_res=self._process_res)
                out.append(np.asarray(pred.depth[0], dtype=np.float32))
        canonical = np.stack(out)
        return np.asarray(
            canonical * (focal_px[:, None, None] / METRIC_CANONICAL_FOCAL), dtype=np.float32
        )
