"""Rotation helpers (SO(3)) in plain numpy, shared by SLAM and reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Mat3 = NDArray[np.float64]
Vec3 = NDArray[np.float64]

_EPS = 1e-9
_SMALL_ANGLE = 1e-6
_NEAR_PI = 1e-4
_PARALLEL = 1e-6


def hat(w: Vec3) -> Mat3:
    """Skew-symmetric matrix such that ``hat(w) @ v == cross(w, v)``."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp(w: Vec3) -> Mat3:
    """Rotation matrix for a rotation vector (axis * angle in radians)."""
    theta = float(np.linalg.norm(w))
    k = hat(w)
    if theta < _EPS:
        return np.asarray(np.eye(3) + k, dtype=np.float64)
    r = np.eye(3) + np.sin(theta) / theta * k + (1 - np.cos(theta)) / theta**2 * (k @ k)
    return np.asarray(r, dtype=np.float64)


def log(r: Mat3) -> Vec3:
    """Rotation vector of a rotation matrix (inverse of ``exp``)."""
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos))
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < _SMALL_ANGLE:
        return np.asarray(v / 2.0, dtype=np.float64)
    if np.pi - theta < _NEAR_PI:
        # Near 180 degrees the antisymmetric part vanishes; use the diagonal.
        axis = np.sqrt(np.clip((np.diag(r) + 1.0) / 2.0, 0.0, None))
        axis *= np.sign(v) + (v == 0)
        return np.asarray(axis / np.linalg.norm(axis) * theta, dtype=np.float64)
    return np.asarray(theta / (2.0 * np.sin(theta)) * v, dtype=np.float64)


def project(r: Mat3) -> Mat3:
    """Nearest rotation matrix to ``r`` (orthonormalise with SVD)."""
    u, _, vt = np.linalg.svd(r)
    d = np.sign(np.linalg.det(u @ vt))
    return np.asarray(u @ np.diag([1.0, 1.0, d]) @ vt, dtype=np.float64)


def slerp(r0: Mat3, r1: Mat3, alpha: float) -> Mat3:
    """Interpolate between two rotations; ``alpha`` in [0, 1]."""
    return np.asarray(r0 @ exp(alpha * log(r0.T @ r1)), dtype=np.float64)


def between(a: Vec3, b: Vec3) -> Mat3:
    """Smallest rotation taking direction ``a`` onto direction ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-9:
        # Opposite vectors: rotate 180 degrees about any axis orthogonal to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < _PARALLEL:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return exp(axis / np.linalg.norm(axis) * np.pi)
    k = hat(v)
    return np.asarray(np.eye(3) + k + k @ k / (1.0 + c), dtype=np.float64)


def kabsch(src: NDArray[np.float64], dst: NDArray[np.float64]) -> Mat3:
    """Rotation ``R`` minimising ``sum |dst_i - R src_i|^2`` for (N, 3) arrays."""
    h = src.T @ dst
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return np.asarray(vt.T @ np.diag([1.0, 1.0, d]) @ u.T, dtype=np.float64)
