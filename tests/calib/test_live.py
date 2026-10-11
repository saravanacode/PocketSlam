import time
from pathlib import Path

import numpy as np
import pytest

from pocketslam.calib.live import LiveCalibration, view_side
from pocketslam.calib.store import ScaleCalibration
from pocketslam.core import so3

SIZE_M = 0.16
FOCAL = 870.0
W, H = 720, 1280
FRAME = np.zeros((H, W, 3), np.uint8)


def project(rotvec: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Corner pixels of the marker for one camera-relative pose."""
    half = SIZE_M / 2
    square = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    cam = square @ so3.exp(rotvec).T + offset
    return np.stack(
        [cam[:, 0] / cam[:, 2] * FOCAL + W / 2, cam[:, 1] / cam[:, 2] * FOCAL + H / 2], 1
    )


def tilted_views() -> list[np.ndarray]:
    """Views from the left, right, above and below, at a few distances and positions."""
    views = []
    for i in range(20):
        angle = np.radians(32 + 3 * (i % 4))
        axis = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)][i % 4]
        offset = np.array([0.05 * np.cos(i), 0.09 * np.sin(i * 0.7), 0.42 + 0.02 * (i % 5)])
        views.append(project(np.array(axis) * angle, offset))
    return views


def feed(cal: LiveCalibration, corners: np.ndarray | None, box: list) -> dict:
    """Submit one frame whose detection is ``corners`` and wait until it is processed."""
    box[0] = corners
    before = box[1]
    cal.submit(FRAME)
    for _ in range(200):
        if box[1] > before or not cal.active:
            break
        time.sleep(0.005)
    return cal.snapshot()


def make(
    tmp_path: Path, size: float | None = SIZE_M
) -> tuple[LiveCalibration, ScaleCalibration, list]:
    box: list = [None, 0]  # [next detection, frames processed]

    def detector(_image: np.ndarray) -> tuple[int, np.ndarray] | None:
        box[1] += 1
        return None if box[0] is None else (18, box[0])

    store = ScaleCalibration(tmp_path / "scale.json")
    return LiveCalibration(store, size, 18, "DICT_4X4_50", detector=detector), store, box


def test_view_side_from_foreshortening() -> None:
    head_on = project(np.zeros(3), np.array([0, 0, 0.5]))
    assert view_side(head_on) is None
    sides = {
        view_side(project(np.array(axis) * 0.6, np.array([0, 0, 0.45])))
        for axis in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    }
    assert sides == {"left", "right", "above", "below"}


def test_guides_then_saves_focal_length(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    cal, store, box = make(tmp_path)
    updates: list[dict] = []
    cal.on_update = updates.append
    cal.start("CPH2653")

    assert feed(cal, None, box)["hint"] == "Point the phone at the marker."
    far = feed(cal, project(np.zeros(3), np.array([0, 0, 2.5])), box)
    assert "Move closer" in far["hint"]
    assert far["views"] == 0  # too small to use

    snap = far
    for corners in tilted_views():
        snap = feed(cal, corners, box)
        if snap["saved"]:
            break
    assert snap["saved"], snap
    assert not snap["active"]
    assert snap["focal_px"] == pytest.approx(FOCAL, rel=0.02)
    assert snap["fov_deg"] == pytest.approx(72.7, abs=2.0)
    assert snap["hint"].startswith("Saved")
    saved = store.focal_ratio("CPH2653")
    assert saved is not None
    assert saved[0] == pytest.approx(FOCAL / H, rel=0.02)
    assert updates
    assert updates[-1]["saved"]
    cal.stop()


def test_head_on_views_never_save(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    cal, store, box = make(tmp_path)
    cal.start("phone")
    snap = {}
    for i in range(16):
        offset = np.array([0.1 * np.cos(i), 0.15 * np.sin(i), 0.45])
        snap = feed(cal, project(np.array([0.0, 0.0, 0.02 * i]), offset), box)
    assert not snap["saved"]
    assert snap["views"] >= 10
    assert "Move the phone" in snap["hint"]  # asks for a tilted view
    assert store.focal_ratio("phone") is None
    cal.stop()
    assert not cal.active


def test_needs_marker_size(tmp_path: Path) -> None:
    cal, _, _ = make(tmp_path, size=None)
    with pytest.raises(RuntimeError, match="Marker size is not set"):
        cal.start("phone")


def test_submit_when_inactive_is_ignored(tmp_path: Path) -> None:
    cal, _, box = make(tmp_path)
    cal.submit(FRAME)
    assert box[1] == 0
    assert cal.snapshot()["active"] is False
