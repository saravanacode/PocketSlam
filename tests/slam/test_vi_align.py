"""Visual-inertial alignment on a synthetic hand-held trajectory with known answers."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pocketslam.core import so3
from pocketslam.slam.vi_align import (
    GRAVITY,
    R_CD_PORTRAIT,
    align_scale_gravity,
    estimate_camera_from_device,
    estimate_time_offset,
    gravity_from_accel,
    preintegrate,
    samples_in_clip,
)

TRUE_SCALE = 2.5
G_WORLD_DIR = np.array([0.3, -0.2, -0.93]) / np.linalg.norm([0.3, -0.2, -0.93])


def test_preintegration_matches_kinematics(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory()
    t0, t1 = clip.times[10], clip.times[11]
    alpha, beta, r_rel = preintegrate(t0, t1, clip.gyro, clip.accel)
    g = GRAVITY * G_WORLD_DIR
    dt = t1 - t0
    r0 = clip.rotation(t0)
    pos = clip.position
    v0 = (pos(np.array(t0 + 1e-5)) - pos(np.array(t0 - 1e-5))) / 2e-5
    expected_p = pos(np.array(t1)) - pos(np.array(t0)) - v0 * dt - 0.5 * g * dt**2
    np.testing.assert_allclose(r0 @ alpha, expected_p, atol=2e-4)
    np.testing.assert_allclose(r0 @ r_rel, clip.rotation(t1), atol=1e-3)
    assert np.linalg.norm(beta) > 0


def test_recovers_scale_and_gravity(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory(noise=0.01)
    result = align_scale_gravity(clip.times, clip.centers, clip.r_wd, clip.gyro, clip.accel)
    assert result.observable, result.reason
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.02)
    angle = np.degrees(np.arccos(np.dot(result.gravity / GRAVITY, G_WORLD_DIR)))
    assert angle < 1.0
    assert np.linalg.norm(result.gravity) == pytest.approx(GRAVITY)


def test_recovers_time_offset(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory(imu_offset=0.04)
    offset, corr = estimate_time_offset(clip.times, clip.r_wc, clip.gyro)
    assert offset == pytest.approx(0.04, abs=0.005)
    assert corr > 0.9


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
def test_picks_phone_orientation(quarter: int, clip_factory: Callable[..., Any]) -> None:
    r_cd = so3.exp(np.array([0.0, 0.0, quarter * np.pi / 2])) @ R_CD_PORTRAIT
    clip = clip_factory(r_cd=r_cd)
    found, err = estimate_camera_from_device(clip.times, clip.r_wc, clip.gyro)
    np.testing.assert_allclose(found, r_cd, atol=1e-9)
    assert err < 0.05


def test_still_phone_is_not_observable_but_gives_gravity(clip_factory: Callable[..., Any]) -> None:
    clip = clip_factory(still=True, noise=0.01)
    result = align_scale_gravity(clip.times, clip.centers, clip.r_wd, clip.gyro, clip.accel)
    assert not result.observable
    assert "accelerated" in result.reason
    g_dir = gravity_from_accel(clip.times, clip.r_wd, clip.accel)
    assert np.degrees(np.arccos(np.dot(g_dir, G_WORLD_DIR))) < 1.0


def test_gravity_is_none_when_imu_misses_the_clip(clip_factory: Callable[..., Any]) -> None:
    """Regression: IMU stamped 5 s after the frames gave a NaN gravity and a NaN point cloud."""
    clip = clip_factory()
    late = clip.accel.shifted(50.0)
    assert samples_in_clip(clip.times, late) == 0
    assert gravity_from_accel(clip.times, clip.r_wd, late) is None
    result = align_scale_gravity(clip.times, clip.centers, clip.r_wd, clip.gyro.shifted(50.0), late)
    assert not result.observable
    assert np.isfinite(result.accel_std)
