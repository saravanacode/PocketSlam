import math

import numpy as np
import pytest

from pocketslam.core.types import Intrinsics, Pose


def test_from_fov_90_degrees_gives_focal_half_width() -> None:
    k = Intrinsics.from_fov(640, 480, 90.0)
    assert k.fx == pytest.approx(320.0)
    assert (k.cx, k.cy) == (320.0, 240.0)
    assert k.source == "fov-estimate"


@pytest.mark.parametrize("fov", [0.0, 180.0, -5.0])
def test_from_fov_rejects_bad_fov(fov: float) -> None:
    with pytest.raises(ValueError, match="hfov_deg"):
        Intrinsics.from_fov(640, 480, fov)


def test_scaled_follows_resolution_change() -> None:
    k = Intrinsics(500.0, 500.0, 320.0, 240.0, 640, 480, source="device")
    half = k.scaled(320, 240)
    assert (half.fx, half.cx, half.cy) == (250.0, 160.0, 120.0)
    assert half.source == "device"
    assert k.scaled(640, 480) is k


def test_matrix_layout() -> None:
    k = Intrinsics(1.0, 2.0, 3.0, 4.0, 10, 10, source="device")
    np.testing.assert_array_equal(k.matrix(), [[1, 0, 3], [0, 2, 4], [0, 0, 1]])


def test_identity_pose_to_tum() -> None:
    assert (
        Pose(timestamp_s=1.5).to_tum()
        == "1.500000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000"
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_to_tum_quaternion_for_180_degree_rotation(axis: int) -> None:
    # Trace is -1 here, which exercises the non-trace branches.
    m = np.eye(4)
    m[:3, :3] = -np.eye(3)
    m[axis, axis] = 1.0
    q = [float(v) for v in Pose(0.0, m).to_tum().split()[4:]]
    expected = [0.0, 0.0, 0.0, 0.0]
    expected[axis] = 1.0
    assert q == pytest.approx(expected)
    assert math.isclose(sum(v * v for v in q), 1.0)
