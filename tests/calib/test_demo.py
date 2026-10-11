import time
from pathlib import Path

import numpy as np
import pytest

from pocketslam.calib.demo import DistanceDemo
from pocketslam.calib.distance import LiveDistance
from pocketslam.calib.live import LiveCalibration
from pocketslam.calib.store import ScaleCalibration, saved_focal
from pocketslam.core import so3

SIZE_M = 0.16
FOCAL = 870.0
W, H = 720, 1280
FRAME = np.zeros((H, W, 3), np.uint8)


def project(rotvec: np.ndarray, offset: np.ndarray) -> np.ndarray:
    half = SIZE_M / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    cam = square @ (so3.exp(rotvec) @ np.diag([1.0, -1.0, -1.0])).T + offset
    u = cam[:, 0] / cam[:, 2] * FOCAL + W / 2
    v = cam[:, 1] / cam[:, 2] * FOCAL + H / 2
    return np.stack([u, v], axis=1) + 0.5


def make(tmp_path: Path) -> tuple[DistanceDemo, ScaleCalibration, list]:
    box: list = [None, 0]

    def detector(_image: np.ndarray) -> tuple[int, np.ndarray] | None:
        box[1] += 1
        return None if box[0] is None else (18, box[0])

    store = ScaleCalibration(tmp_path / "scale.json")
    calibration = LiveCalibration(store, SIZE_M, 18, "DICT_4X4_50", detector=detector)
    distance = LiveDistance(SIZE_M, 18, "DICT_4X4_50", detector=detector)
    return DistanceDemo(calibration, distance), store, box


def feed(demo: DistanceDemo, corners: np.ndarray | None, box: list) -> None:
    box[0] = corners
    before = box[1]
    demo.submit(FRAME)
    for _ in range(200):
        if box[1] > before:
            break
        time.sleep(0.005)
    time.sleep(0.01)


def test_calibrated_phone_goes_straight_to_measuring(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    demo, store, box = make(tmp_path)
    store.add_focal("CPH2649", FOCAL / H, 0.017, "live")
    assert demo.stage == "idle"
    assert demo.start("CPH2649") == "measuring"
    for _ in range(5):
        feed(demo, project(np.zeros(3), np.array([0, 0, 1.2])), box)
    snap = demo.snapshot()
    assert snap["stage"] == "measuring"
    assert snap["distance"]["distance_m"] == pytest.approx(1.2, rel=0.01)
    demo.stop()
    assert demo.stage == "idle"


def test_uncalibrated_phone_calibrates_then_measures_by_itself(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    demo, store, box = make(tmp_path)
    assert demo.start("new-phone") == "calibrating"
    for i in range(30):
        axis = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)][i % 4]
        offset = np.array([0.05 * np.cos(i), 0.09 * np.sin(i * 0.7), 0.42 + 0.02 * (i % 5)])
        feed(demo, project(np.array(axis) * np.radians(32 + 3 * (i % 4)), offset), box)
        if demo.stage == "measuring":
            break
    assert demo.stage == "measuring"
    saved = saved_focal(store, "new-phone")
    assert saved is not None
    assert saved[0] == pytest.approx(FOCAL / H, rel=0.02)
    for _ in range(5):
        feed(demo, project(np.zeros(3), np.array([0, 0, 0.8])), box)
    assert demo.snapshot()["distance"]["distance_m"] == pytest.approx(0.8, rel=0.03)
    demo.stop()


def test_recalibrate_forces_calibration(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    demo, store, _ = make(tmp_path)
    store.add_focal("p", FOCAL / H, 0.017, "live")
    assert demo.start("p", recalibrate=True) == "calibrating"
    demo.stop()
    assert demo.stage == "idle"
    assert not demo.calibration.active


def test_plain_calibration_does_not_start_measuring(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    demo, _, box = make(tmp_path)
    demo.calibration.start("p")  # the separate "Calibrate camera" button
    for i in range(30):
        axis = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)][i % 4]
        offset = np.array([0.05 * np.cos(i), 0.09 * np.sin(i * 0.7), 0.42 + 0.02 * (i % 5)])
        feed(demo, project(np.array(axis) * np.radians(32 + 3 * (i % 4)), offset), box)
        if not demo.calibration.active:
            break
    assert demo.calibration.snapshot()["saved"]
    assert demo.stage == "idle"


def test_missing_marker_size_is_reported(tmp_path: Path) -> None:
    store = ScaleCalibration(tmp_path / "scale.json")
    demo = DistanceDemo(
        LiveCalibration(store, None, None, "DICT_4X4_50", detector=lambda _i: None),
        LiveDistance(None, None, "DICT_4X4_50", detector=lambda _i: None),
    )
    with pytest.raises(RuntimeError, match="Marker size is not set"):
        demo.start("p")
    assert demo.stage == "idle"
