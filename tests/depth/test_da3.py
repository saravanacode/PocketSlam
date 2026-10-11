from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pocketslam.core.device import select_device
from pocketslam.depth.da3 import METRIC_CANONICAL_FOCAL, DA3Metric, DA3MultiView, load_model

DA3_SMALL = ("depth-anything/DA3-SMALL", "e08cab65ca0ec38e7826075418411ab90cab4da3")


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def inference(self, image: list[np.ndarray], **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"n": len(image), **kwargs})
        n = len(image)
        ext = np.tile(np.eye(4)[:3], (n, 1, 1))
        ext[:, 0, 3] = np.arange(n)
        return SimpleNamespace(
            depth=np.full((n, 6, 8), 2.0),
            conf=np.ones((n, 6, 8)),
            intrinsics=np.tile(np.array([[10.0, 0, 4], [0, 10.0, 3], [0, 0, 1]]), (n, 1, 1)),
            extrinsics=ext,
            processed_images=np.zeros((n, 6, 8, 3), dtype=np.uint8),
        )


def test_multiview_wraps_prediction() -> None:
    model = FakeModel()
    geo = DA3MultiView(model, process_res=320).estimate_views([np.zeros((6, 8, 3), np.uint8)] * 3)
    assert model.calls == [{"n": 3, "process_res": 320}]
    assert len(geo) == 3
    assert geo.depth.dtype == np.float32
    assert not geo.is_metric
    np.testing.assert_allclose(geo.world_to_camera[2, :, 3], [2, 0, 0, 1])
    np.testing.assert_allclose(geo.camera_to_world()[2, 0, 3], -2.0)


def test_multiview_needs_two_frames() -> None:
    with pytest.raises(ValueError, match="two frames"):
        DA3MultiView(FakeModel()).estimate_views([np.zeros((6, 8, 3), np.uint8)])


def test_metric_depth_uses_focal_scaling() -> None:
    model = FakeModel()
    depth = DA3Metric(model).metric_depth(
        [np.zeros((6, 8, 3), np.uint8)] * 2, np.array([600.0, 300.0])
    )
    assert len(model.calls) == 2  # one frame per call
    assert depth[0, 0, 0] == pytest.approx(2.0 * 600.0 / METRIC_CANONICAL_FOCAL)
    assert depth[1, 0, 0] == pytest.approx(2.0)


@pytest.mark.slow
def test_real_small_model_runs_on_this_machine() -> None:
    pytest.importorskip("depth_anything_3")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    frames = [np.roll(base, shift, axis=1) for shift in (0, 8, 16, 24)]
    model = load_model("depth-anything/DA3-SMALL", "main", select_device(None))
    geo = DA3MultiView(model, process_res=252).estimate_views(frames)
    assert len(geo) == 4
    assert np.isfinite(geo.depth).all()
    assert geo.world_to_camera.shape == (4, 4, 4)
