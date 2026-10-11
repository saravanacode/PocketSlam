import numpy as np
import pytest

from pocketslam.core import so3

rng = np.random.default_rng(0)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.3, 2.0, np.pi - 1e-6])
def test_exp_log_round_trip(angle: float) -> None:
    axis = np.array([0.3, -0.5, 0.8])
    w = axis / np.linalg.norm(axis) * angle
    r = so3.exp(w)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(so3.exp(so3.log(r)), r, atol=1e-6)


def test_slerp_midpoint() -> None:
    r1 = so3.exp(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(
        so3.slerp(np.eye(3), r1, 0.5), so3.exp(np.array([0, 0, 0.5])), atol=1e-12
    )


@pytest.mark.parametrize("b", [[0, 0, 1], [0, 0, -1], [1, 2, 3]])
def test_between_maps_direction(b: list[float]) -> None:
    a = np.array([0.0, 0.0, 1.0])
    target = np.asarray(b, dtype=float)
    r = so3.between(a, target)
    np.testing.assert_allclose(r @ a, target / np.linalg.norm(target), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-9)


def test_kabsch_recovers_rotation() -> None:
    r = so3.exp(np.array([0.2, -0.4, 0.9]))
    src = rng.normal(size=(50, 3))
    np.testing.assert_allclose(so3.kabsch(src, src @ r.T), r, atol=1e-9)


def test_project_returns_rotation() -> None:
    noisy = so3.exp(np.array([0.1, 0.2, 0.3])) + 0.01 * rng.normal(size=(3, 3))
    r = so3.project(noisy)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
