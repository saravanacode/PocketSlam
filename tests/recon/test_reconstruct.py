from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pocketslam.calib.store import ScaleCalibration, device_label
from pocketslam.config import ReconSettings
from pocketslam.core.types import MultiViewGeometry
from pocketslam.recon.reader import Recording
from pocketslam.recon.reconstruct import (
    metric_scale,
    reconstruct,
    select_keyframes,
)
from pocketslam.slam.vi_align import ImuSeries

TRUE_SCALE = 2.5
G_WORLD_DIR = np.array([0.3, -0.2, -0.93]) / np.linalg.norm([0.3, -0.2, -0.93])
K = np.array([[20.0, 0, 8], [0, 20.0, 6], [0, 0, 1]])


class FakeMultiView:
    """Returns the synthetic clip's poses (model units) and a plane of depth 1."""

    def __init__(self, clip: Any) -> None:
        self.clip = clip

    def estimate_views(self, images: Sequence[np.ndarray]) -> MultiViewGeometry:
        n = len(images)
        c2w = np.tile(np.eye(4), (n, 1, 1))
        c2w[:, :3, :3] = self.clip.r_wc[:n]
        c2w[:, :3, 3] = self.clip.centers[:n]
        return MultiViewGeometry(
            depth=np.ones((n, 12, 16), np.float32),
            confidence=np.ones((n, 12, 16), np.float32),
            intrinsics=np.tile(K, (n, 1, 1)),
            world_to_camera=np.linalg.inv(c2w),
            images=np.zeros((n, 12, 16, 3), np.uint8),
            is_metric=False,
        )


class FakeMetric:
    def __init__(self, metres_per_unit: float) -> None:
        self.k = metres_per_unit

    def metric_depth(self, images: Sequence[np.ndarray], focal_px: np.ndarray) -> np.ndarray:
        return np.full((len(images), 12, 16), self.k, np.float32)


NO_IMU = ImuSeries(np.zeros(0), np.zeros((0, 3)))


def recording_from(clip: Any, imu: bool = True) -> Recording:
    """A Recording whose frames sit exactly on the clip's keyframe times."""
    rec = Recording(
        path=Path("synthetic.mcap"),
        frame_times=clip.times,
        frame_jpegs=[b""] * len(clip.times),
        gyro=clip.gyro if imu else NO_IMU,
        accel=clip.accel if imu else NO_IMU,
    )
    rec.decode = lambda i: np.zeros((12, 16, 3), np.uint8)  # type: ignore[method-assign]
    return rec


SETTINGS = ReconSettings(
    max_frames=200,
    keyframe_fps=10.0,
    conf_percentile=0.0,
    voxel_m=0.05,
    _env_file=None,  # type: ignore[call-arg]
)


def test_select_keyframes_spacing_and_cap() -> None:
    times = np.arange(0, 5, 1 / 30)
    idx = select_keyframes(times, fps=10, max_frames=1000)
    assert np.all(np.diff(times[idx]) >= 0.1 - 1e-6)
    assert len(select_keyframes(times, fps=10, max_frames=8)) == 8


def test_imu_scale_and_gravity_alignment(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory(noise=0.01)
    result = reconstruct(recording_from(clip), FakeMultiView(clip), FakeMetric(2.4), SETTINGS)
    assert result.scale_source == "imu"
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.03)
    up_in_output = result.rotation @ (-G_WORLD_DIR)
    np.testing.assert_allclose(up_in_output, [0, 0, 1], atol=0.02)
    assert result.summary["imu"]["observable"]
    assert result.summary["scale_disagreement"] == pytest.approx(
        abs(result.scale - 2.4) / 2.4, abs=1e-3
    )
    assert len(result.cloud) > 0


def test_falls_back_to_metric_scale_without_imu(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory()
    result = reconstruct(
        recording_from(clip, imu=False), FakeMultiView(clip), FakeMetric(1.7), SETTINGS
    )
    assert result.scale_source == "metric-depth"
    assert result.scale == pytest.approx(1.7)
    assert not result.summary["gravity_aligned"]


def test_no_scale_source_keeps_model_units(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory()
    result = reconstruct(recording_from(clip, imu=False), FakeMultiView(clip), None, SETTINGS)
    assert (result.scale, result.scale_source) == (1.0, "none")


def test_metric_scale_is_median_ratio() -> None:
    n = 2
    g = MultiViewGeometry(
        depth=np.full((n, 4, 4), 2.0, np.float32),
        confidence=np.ones((n, 4, 4), np.float32),
        intrinsics=np.tile(K, (n, 1, 1)),
        world_to_camera=np.tile(np.eye(4), (n, 1, 1)),
        images=np.zeros((n, 4, 4, 3), np.uint8),
        is_metric=False,
    )
    assert metric_scale(g, np.full((n, 4, 4), 3.0, np.float32), 0.0) == pytest.approx(1.5)


# A marker 8 px wide at depth 1 with f = 20 is 0.4 model units; at 0.16 m real
# size the scale is 0.4.
MARKER_CORNERS = np.array([[4.0, 2.0], [12.0, 2.0], [12.0, 10.0], [4.0, 10.0]])
MARKER_SETTINGS = SETTINGS.model_copy(update={"marker_size_m": 0.16})


def detect_always(_image: np.ndarray) -> tuple[int, np.ndarray]:
    return 18, MARKER_CORNERS


def test_marker_scale_wins_and_is_remembered(
    clip_factory: Callable[..., Any], tmp_path: Path
) -> None:
    clip = clip_factory(noise=0.01)
    calibration = ScaleCalibration(tmp_path / "scale.json")
    # Focal length measured for this phone: 1.25 x the longer side (16 px) = 20 px,
    # the same as the fake model's own value.
    calibration.add_focal("unknown", 1.25, 0.02, "calibration-clip")
    result = reconstruct(
        recording_from(clip),
        FakeMultiView(clip),
        FakeMetric(2.0),
        MARKER_SETTINGS,
        detect=detect_always,
        calibration=calibration,
    )
    assert result.scale_source == "marker"
    assert result.summary["marker"]["focal_calibrated"]
    assert result.scale == pytest.approx(0.4, rel=1e-3)
    marker = result.summary["marker"]
    assert marker["views_used"] == marker["views_found"] == len(clip.times)
    assert marker["metric_depth_ratio"] == pytest.approx(0.2, rel=1e-3)
    # Output corners are in metres. This fake returns the same marker pixels
    # while the camera moves, so only the average size is meaningful here;
    # exact corners are checked in test_marker.py with a consistent scene.
    corners = np.array(marker["corners_world"])
    assert corners.shape == (4, 3)
    edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    assert edges.mean() == pytest.approx(0.16, rel=0.1)
    assert calibration.ratio("unknown") == (pytest.approx(0.2, rel=1e-3), 1, True)

    # A later clip without the marker reuses the correction.
    later = reconstruct(
        recording_from(clip, imu=False),
        FakeMultiView(clip),
        FakeMetric(2.0),
        MARKER_SETTINGS,
        detect=lambda _image: None,
        calibration=calibration,
    )
    assert later.scale_source == "metric-depth (calibrated)"
    assert later.scale == pytest.approx(0.4, rel=1e-3)
    assert later.summary["calibration"]["samples"] == 1
    assert later.summary["marker"] is None


def test_marker_in_one_view_is_not_enough(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory()
    seen = iter([True])

    def detect_once(_image: np.ndarray) -> tuple[int, np.ndarray] | None:
        return (18, MARKER_CORNERS) if next(seen, False) else None

    result = reconstruct(
        recording_from(clip, imu=False),
        FakeMultiView(clip),
        FakeMetric(1.7),
        MARKER_SETTINGS,
        detect=detect_once,
    )
    assert result.scale_source == "metric-depth"
    assert result.scale == pytest.approx(1.7)


def test_marker_ignored_when_size_not_set(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory()
    result = reconstruct(
        recording_from(clip, imu=False), FakeMultiView(clip), None, SETTINGS, detect=detect_always
    )
    assert result.scale_source == "none"
    assert "marker" not in result.summary


def test_calibrated_focal_replaces_model_estimate(
    clip_factory: Callable[..., Any], tmp_path: Path
) -> None:
    clip = clip_factory()
    calibration = ScaleCalibration(tmp_path / "scale.json")
    calibration.add_focal("unknown", 2.5, 0.02, "clip")  # f = 2.5 x longer side (16 px) = 40 px
    result = reconstruct(
        recording_from(clip, imu=False),
        FakeMultiView(clip),
        None,
        MARKER_SETTINGS,
        detect=detect_always,
        calibration=calibration,
    )
    assert result.summary["focal"] == {
        "model_estimate_px": 20.0,
        "calibrated_px": 40.0,
        "ratio": 2.0,
        "at_resolution": [12, 16],
        "samples": 1,
    }
    np.testing.assert_allclose(result.geometry.intrinsics[:, 0, 0], 40.0)
    # Twice the focal length halves the marker's size in model units, doubling the scale.
    assert result.scale == pytest.approx(0.8, rel=1e-3)


def test_device_label_prefers_model() -> None:
    assert device_label({"model": "CPH2653", "device": "Mozilla/5.0 K"}) == "CPH2653"
    assert device_label({"model": "", "device": "Mozilla/5.0 K"}) == "Mozilla/5.0 K"
    assert device_label(None) == "unknown"


def test_marker_correction_not_remembered_without_measured_focal(
    clip_factory: Callable[..., Any], tmp_path: Path
) -> None:
    clip = clip_factory()
    calibration = ScaleCalibration(tmp_path / "scale.json")
    result = reconstruct(
        recording_from(clip, imu=False),
        FakeMultiView(clip),
        FakeMetric(2.0),
        MARKER_SETTINGS,
        detect=detect_always,
        calibration=calibration,
    )
    assert result.scale_source == "marker"  # still used for this scan
    assert not result.summary["marker"]["focal_calibrated"]
    assert calibration.ratio("unknown") is None  # but not remembered


def test_imu_that_does_not_overlap_frames_is_ignored(clip_factory: Callable[..., Any]) -> None:
    """Regression: a lagging video left the IMU 5 s from the frames; all points were NaN."""
    clip = clip_factory()
    rec = recording_from(clip)
    rec.gyro, rec.accel = clip.gyro.shifted(5.0 + 5.0), clip.accel.shifted(5.0 + 5.0)
    result = reconstruct(rec, FakeMultiView(clip), FakeMetric(1.7), SETTINGS)
    assert result.scale_source == "metric-depth"
    assert not result.summary["gravity_aligned"]
    assert "do not overlap" in result.summary["imu"]["reason"]
    assert np.isfinite(result.cloud.xyz).all()
    assert len(result.cloud) > 1
    assert np.isfinite(result.summary["path_length_m"])
