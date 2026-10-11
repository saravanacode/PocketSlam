from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pocketslam.config import Settings
from pocketslam.core import so3
from pocketslam.core.types import Frame, Intrinsics
from pocketslam.slam.vi_align import GRAVITY, R_CD_PORTRAIT, ImuSeries


@pytest.fixture
def make_frame() -> Callable[..., Frame]:
    def _make(width: int = 64, height: int = 48, timestamp_s: float = 0.0, index: int = 0) -> Frame:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return Frame(image, timestamp_s, index, Intrinsics.from_fov(width, height, 60.0))

    return _make


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        allow_insecure=True,
        output_dir=tmp_path / "outputs",
        web_dir=Path(__file__).resolve().parents[1] / "web",
        _env_file=None,  # type: ignore[call-arg]
    )


# Synthetic hand-held clip with known scale, gravity and IMU (used by SLAM and recon tests).
TRUE_SCALE = 2.5
G_WORLD_DIR = np.array([0.3, -0.2, -0.93]) / np.linalg.norm([0.3, -0.2, -0.93])
R_WORLD = so3.exp(np.array([0.4, -0.7, 1.1]))  # arbitrary visual world orientation


def position(t: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            0.25 * np.sin(2 * np.pi * 0.45 * t),
            0.12 * np.sin(2 * np.pi * 0.8 * t + 0.5),
            0.08 * np.cos(2 * np.pi * 0.6 * t),
        ],
        axis=-1,
    )


def rotation(t: float) -> np.ndarray:
    w = np.array([0.35 * np.sin(1.3 * t), 0.25 * np.sin(0.9 * t + 1.0), 0.45 * np.sin(0.7 * t)])
    return R_WORLD @ so3.exp(w)


@dataclass
class Clip:
    times: np.ndarray
    centers: np.ndarray
    r_wc: np.ndarray
    r_wd: np.ndarray
    gyro: ImuSeries
    accel: ImuSeries
    position: Callable[[np.ndarray], np.ndarray] = position
    rotation: Callable[[float], np.ndarray] = rotation


def make_clip(
    duration: float = 5.0,
    imu_offset: float = 0.0,
    noise: float = 0.0,
    still: bool = False,
    r_cd: np.ndarray = R_CD_PORTRAIT,
) -> Clip:
    rng = np.random.default_rng(1)
    g_world = GRAVITY * G_WORLD_DIR
    h = 1e-4
    t_imu = np.arange(0.0, duration, 0.01)
    gyro, accel = [], []
    for t in t_imu:
        r = rotation(t)
        w_dev = so3.log(r.T @ rotation(t + h)) / h
        if still:
            acc_w = np.zeros(3)
        else:
            acc_w = (
                position(np.array(t + h)) - 2 * position(np.array(t)) + position(np.array(t - h))
            ) / h**2
        gyro.append(w_dev + noise * rng.normal(size=3))
        accel.append(r.T @ (acc_w - g_world) + noise * rng.normal(size=3))
    kf = np.arange(0.05, duration - 0.05, 0.1)
    r_wd = np.stack([rotation(t) for t in kf])
    r_wc = np.stack([r @ r_cd.T for r in r_wd])
    pos = np.zeros((len(kf), 3)) if still else position(kf)
    # Recorded IMU stamps run `imu_offset` behind the camera clock.
    return Clip(
        times=kf,
        centers=pos / TRUE_SCALE,
        r_wc=r_wc,
        r_wd=r_wd,
        gyro=ImuSeries(t_imu - imu_offset, np.array(gyro)),
        accel=ImuSeries(t_imu - imu_offset, np.array(accel)),
    )


@pytest.fixture
def clip_factory() -> Callable[..., Clip]:
    return make_clip
