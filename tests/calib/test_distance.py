import time

import numpy as np
import pytest

from pocketslam.calib.distance import LiveDistance, distance_error, marker_pose
from pocketslam.core import so3

SIZE_M = 0.16
FOCAL = 870.0
W, H = 720, 1280
FRAME = np.zeros((H, W, 3), np.uint8)


def project(
    distance: float, tilt_deg: float = 0.0, offset: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Corner pixels of the marker at ``distance`` along the view axis, tilted about y."""
    half = SIZE_M / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    r = so3.exp(np.array([0.0, np.radians(tilt_deg), 0.0])) @ np.diag([1.0, -1.0, -1.0])
    cam = square @ r.T + np.array([offset[0], offset[1], distance])
    return (
        np.stack([cam[:, 0] / cam[:, 2] * FOCAL + W / 2, cam[:, 1] / cam[:, 2] * FOCAL + H / 2], 1)
        + 0.5
    )


@pytest.mark.parametrize(("distance", "tilt"), [(0.3, 0), (1.0, 25), (2.5, 40), (5.0, 10)])
def test_pose_recovers_distance_and_tilt(distance: float, tilt: float) -> None:
    pytest.importorskip("cv2")
    pose = marker_pose(project(distance, tilt), FOCAL, (W, H), SIZE_M)
    assert pose is not None
    assert pose.distance_m == pytest.approx(distance, rel=0.005)
    assert pose.tilt_deg == pytest.approx(tilt, abs=1.0)


def test_off_axis_distance_is_to_the_marker_centre() -> None:
    pytest.importorskip("cv2")
    pose = marker_pose(project(1.0, 0, offset=(0.3, 0.4)), FOCAL, (W, H), SIZE_M)
    assert pose is not None
    assert pose.distance_m == pytest.approx(np.sqrt(1 + 0.09 + 0.16), rel=0.005)


def test_error_grows_with_distance() -> None:
    near = distance_error(0.5, side_px=280.0, focal_rel_std=0.017)
    far = distance_error(4.0, side_px=35.0, focal_rel_std=0.017)
    assert near == pytest.approx(0.5 * 0.0174, rel=0.05)
    assert far / 4.0 > near / 0.5  # relative error is larger when the marker is small


def make() -> tuple[LiveDistance, list]:
    box: list = [None, 0]  # [next detection, frames processed]

    def detector(_image: np.ndarray) -> tuple[int, np.ndarray] | None:
        box[1] += 1
        return None if box[0] is None else (18, box[0])

    return LiveDistance(SIZE_M, 18, "DICT_4X4_50", detector=detector), box


def feed(live: LiveDistance, corners: np.ndarray | None, box: list) -> dict:
    box[0] = corners
    before = box[1]
    live.submit(FRAME)
    for _ in range(200):
        if box[1] > before:
            break
        time.sleep(0.005)
    time.sleep(0.005)  # let the worker store the result
    return live.snapshot()


def test_tracks_distance_and_keeps_last_value_when_lost() -> None:
    pytest.importorskip("cv2")
    live, box = make()
    updates: list[dict] = []
    live.on_update = updates.append
    live.start("CPH2649", FOCAL / H, 0.017)
    snap = {}
    for d in (0.5, 0.5, 0.5, 0.5, 0.5):
        snap = feed(live, project(d), box)
    assert snap["marker_visible"]
    assert snap["distance_m"] == pytest.approx(0.5, rel=0.01)
    assert snap["error_m"] == pytest.approx(0.5 * 0.0174, rel=0.2)
    assert snap["focal_px"] == pytest.approx(FOCAL)
    assert len(snap["corners"]) == 4

    for d in (1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0):
        snap = feed(live, project(d), box)
    assert snap["distance_m"] == pytest.approx(2.0, rel=0.01)
    assert (snap["min_m"], snap["max_m"]) == (
        pytest.approx(0.5, rel=0.01),
        pytest.approx(2.0, rel=0.01),
    )

    time.sleep(0.6)
    lost = feed(live, None, box)
    assert not lost["marker_visible"]
    assert lost["distance_m"] == pytest.approx(2.0, rel=0.01)  # last value kept
    assert lost["corners"] is None
    live.stop()
    assert not live.active
    assert updates
    assert updates[-1]["active"] is False


def test_median_ignores_a_single_outlier() -> None:
    pytest.importorskip("cv2")
    live, box = make()
    live.start("p", FOCAL / H, 0.02)
    for d in (1.0, 1.0, 3.0, 1.0, 1.0):
        snap = feed(live, project(d), box)
    assert snap["distance_m"] == pytest.approx(1.0, rel=0.01)
    assert snap["raw_m"] == pytest.approx(1.0, rel=0.01)
    live.stop()


def test_needs_marker_size() -> None:
    live = LiveDistance(None, 18, "DICT_4X4_50", detector=lambda _i: None)
    with pytest.raises(RuntimeError, match="Marker size is not set"):
        live.start("p", 0.8, 0.02)
