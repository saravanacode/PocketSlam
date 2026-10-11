"""Visual-inertial alignment: metric scale and gravity from up-to-scale poses plus IMU.

A monocular model (for example Depth Anything 3) gives camera poses up to an
unknown scale, in an arbitrary world frame. The phone's accelerometer measures
specific force ``a = R_dw (p'' - g)`` in its own (device) frame. Integrating it
between keyframes and matching against the visual trajectory gives the metric
scale ``s`` and the gravity vector ``g`` in the visual world frame, as in the
VINS-Mono initialisation. All functions are plain numpy.

Conventions:

* ``R_wc``: camera-to-world rotation (OpenCV camera axes: x right, y down, z forward).
* ``R_cd``: device-to-camera rotation, ``v_c = R_cd @ v_d``. Android device
  axes: x right, y up, z out of the screen.
* Times are seconds on one clock. ``offset`` means ``camera_time = imu_time + offset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pocketslam.core import so3

GRAVITY = 9.81
# Rear camera, phone held in portrait, image delivered upright.
R_CD_PORTRAIT = np.diag([1.0, -1.0, -1.0])
MIN_ROTATION_RAD_S = 0.15
MIN_ACCEL_STD = 0.2
MAX_SCALE_REL_STD = 0.15
_TINY = 1e-9
_AXIS_ALIGNED = 0.9

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ImuSeries:
    """Time-ordered samples from one IMU sensor.

    Attributes:
        t: ``(M,)`` sample times in seconds.
        xyz: ``(M, 3)`` values in the device frame.
    """

    t: Array
    xyz: Array

    def shifted(self, offset_s: float) -> ImuSeries:
        """The same samples with ``offset_s`` added to every timestamp."""
        return ImuSeries(self.t + offset_s, self.xyz)

    def interp(self, times: Array) -> Array:
        """Linearly interpolated values at ``times``, shape ``(len(times), 3)``."""
        return np.stack([np.interp(times, self.t, self.xyz[:, i]) for i in range(3)], axis=1)

    def mean_over(self, t0: float, t1: float, steps: int = 8) -> Array:
        """Average value over ``[t0, t1]`` from evenly spaced interpolated samples."""
        return np.asarray(self.interp(np.linspace(t0, t1, steps)).mean(axis=0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ViAlignment:
    """Result of the scale and gravity solve.

    Attributes:
        scale: Multiply visual positions and depths by this to get metres.
        gravity: Gravity vector in the visual world frame, metres per second squared.
        velocities: ``(N, 3)`` device velocity at each keyframe, metric, world frame.
        residual_rms: RMS of the least-squares residual.
        scale_rel_std: Standard deviation of ``scale`` divided by ``scale``.
        accel_std: Standard deviation of the accelerometer magnitude (excitation).
        observable: True when the IMU constrains the scale well enough to use it.
        reason: Why the scale is not observable, empty when it is.
    """

    scale: float
    gravity: Array
    velocities: Array
    residual_rms: float
    scale_rel_std: float
    accel_std: float
    observable: bool
    reason: str


def visual_angular_velocity(times: Array, r_wc: Array) -> Array:
    """Mean angular velocity between consecutive keyframes, in the camera frame.

    Args:
        times: ``(N,)`` keyframe times.
        r_wc: ``(N, 3, 3)`` camera-to-world rotations.

    Returns:
        ``(N - 1, 3)`` angular velocity in rad/s, expressed in the earlier camera frame.
    """
    dt = np.diff(times)
    return np.stack([so3.log(r_wc[k].T @ r_wc[k + 1]) / dt[k] for k in range(len(dt))])


def estimate_time_offset(
    times: Array, r_wc: Array, gyro: ImuSeries, search_s: float = 0.2, step_s: float = 0.002
) -> tuple[float, float]:
    """Find the camera-IMU time offset by matching rotation speeds.

    Rotation speed does not depend on how the camera and device frames are
    oriented, so this works before ``R_cd`` is known.

    Args:
        times: ``(N,)`` keyframe times.
        r_wc: ``(N, 3, 3)`` camera-to-world rotations.
        gyro: Gyroscope samples (rad/s).
        search_s: Search the offset in ``[-search_s, search_s]``.
        step_s: Search step.

    Returns:
        ``(offset_s, correlation)``: add ``offset_s`` to IMU times to put them on
        the camera clock. Correlation near 1 means a clear match; below about 0.5
        there was too little rotation to tell.
    """
    visual = np.linalg.norm(visual_angular_velocity(times, r_wc), axis=1)
    best_offset, best_corr = 0.0, -np.inf
    for offset in np.arange(-search_s, search_s + step_s / 2, step_s):
        g = gyro.shifted(float(offset))
        speeds = np.array(
            [np.linalg.norm(g.mean_over(times[k], times[k + 1])) for k in range(len(visual))]
        )
        if np.std(speeds) < _TINY or np.std(visual) < _TINY:
            continue
        corr = float(np.corrcoef(speeds, visual)[0, 1])
        if corr > best_corr:
            best_offset, best_corr = float(offset), corr
    return best_offset, (best_corr if np.isfinite(best_corr) else 0.0)


def estimate_camera_from_device(
    times: Array, r_wc: Array, gyro: ImuSeries, prior: Array = R_CD_PORTRAIT
) -> tuple[Array, float]:
    """Choose ``R_cd`` among the four upright phone orientations.

    The image can be portrait or either landscape, which rotates the camera
    axes about the optical axis by a multiple of 90 degrees. The candidate
    that best maps gyro angular velocity onto visual angular velocity wins.

    Args:
        times: ``(N,)`` keyframe times.
        r_wc: ``(N, 3, 3)`` camera-to-world rotations.
        gyro: Gyroscope samples already on the camera clock.
        prior: Used when the clip has too little rotation to decide.

    Returns:
        ``(R_cd, rms_error)`` with the error in rad/s; NaN when the prior was used.
    """
    visual = visual_angular_velocity(times, r_wc)
    if np.mean(np.linalg.norm(visual, axis=1)) < MIN_ROTATION_RAD_S:
        return prior, float("nan")
    device = np.stack([gyro.mean_over(times[k], times[k + 1]) for k in range(len(visual))])
    best, best_err = prior, np.inf
    for quarter in range(4):
        candidate = so3.exp(np.array([0.0, 0.0, quarter * np.pi / 2])) @ R_CD_PORTRAIT
        err = float(np.sqrt(np.mean(np.sum((visual - device @ candidate.T) ** 2, axis=1))))
        if err < best_err:
            best, best_err = candidate, err
    return best, best_err


def device_rotations_at(times: Array, r_wd: Array, query: Array) -> Array:
    """Device-to-world rotations at ``query`` times by slerp between keyframes."""
    out = np.empty((len(query), 3, 3))
    for i, t in enumerate(query):
        k = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
        alpha = float(np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0))
        out[i] = so3.slerp(r_wd[k], r_wd[k + 1], alpha)
    return out


def preintegrate(
    t0: float, t1: float, gyro: ImuSeries, accel: ImuSeries
) -> tuple[Array, Array, Array]:
    """Integrate IMU between two keyframes in the device frame at ``t0``.

    Args:
        t0: Start time.
        t1: End time.
        gyro: Gyroscope samples (rad/s), on the camera clock.
        accel: Accelerometer samples (m/s^2, with gravity), on the camera clock.

    Returns:
        ``(alpha, beta, r_rel)``: position and velocity change from specific
        force alone, and the rotation from device frame at ``t1`` to ``t0``.
    """
    inside = accel.t[(accel.t > t0) & (accel.t < t1)]
    grid = np.concatenate([[t0], inside, [t1]])
    a = accel.interp(grid)
    w = gyro.interp(grid)
    rot = np.eye(3)
    alpha = np.zeros(3)
    beta = np.zeros(3)
    for k in range(len(grid) - 1):
        dt = grid[k + 1] - grid[k]
        rot_next = rot @ so3.exp(0.5 * (w[k] + w[k + 1]) * dt)
        acc = 0.5 * (rot @ a[k] + rot_next @ a[k + 1])
        alpha += beta * dt + 0.5 * acc * dt * dt
        beta += acc * dt
        rot = rot_next
    return alpha, beta, rot


def samples_in_clip(times: Array, series: ImuSeries) -> int:
    """Number of IMU samples between the first and last keyframe."""
    return int(np.count_nonzero((series.t >= times[0]) & (series.t <= times[-1])))


def gravity_from_accel(times: Array, r_wd: Array, accel: ImuSeries) -> Array | None:
    """Gravity direction (unit vector, world frame) from the mean accelerometer reading.

    Over a short hand-held clip the mean linear acceleration is close to zero,
    so the mean of ``R_wd @ a`` points opposite to gravity. Use this when the
    scale solve is not observable.

    Returns:
        The direction, or None when no accelerometer samples fall inside the
        clip (for example the video lagged behind the IMU) or they cancel out.
    """
    mask = (accel.t >= times[0]) & (accel.t <= times[-1])
    if not mask.any():
        return None
    rots = device_rotations_at(times, r_wd, accel.t[mask])
    mean = np.einsum("nij,nj->i", rots, accel.xyz[mask]) / int(mask.sum())
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm < _TINY:
        return None
    return np.asarray(-mean / norm, dtype=np.float64)


def _tangent_basis(g_dir: Array) -> Array:
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(g_dir[0]) < _AXIS_ALIGNED else np.array([0.0, 1.0, 0.0])
    )
    b1 = np.cross(g_dir, helper)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(g_dir, b1)
    return np.stack([b1, b2], axis=1)


Preintegrated = tuple[Array, Array, Array]


def _build_system(
    times: Array,
    centers: Array,
    r_wd: Array,
    pre: list[Preintegrated],
    *,
    g_dir: Array | None,
    g_norm: float,
) -> tuple[Array, Array]:
    """Stack the position and velocity constraints into ``A x = b``.

    Unknowns: ``[v_0 .. v_{N-1}, gravity, s]``. Gravity is 3 free values when
    ``g_dir`` is None, else a 2-D correction orthogonal to ``g_norm * g_dir``.
    """
    n = len(times)
    basis = None if g_dir is None else _tangent_basis(g_dir)
    g_cols = 3 if basis is None else 2
    cols = 3 * n + g_cols + 1
    gi, si = 3 * n, cols - 1
    g0 = np.zeros(3) if g_dir is None else g_norm * g_dir
    rows, rhs = [], []
    for k in range(n - 1):
        dt = times[k + 1] - times[k]
        alpha, beta, _ = pre[k]
        a_p = np.zeros((3, cols))
        a_p[:, 3 * k : 3 * k + 3] = -dt * np.eye(3)
        a_p[:, si] = centers[k + 1] - centers[k]
        a_p[:, gi : gi + g_cols] = -0.5 * dt * dt * (np.eye(3) if basis is None else basis)
        a_v = np.zeros((3, cols))
        a_v[:, 3 * (k + 1) : 3 * (k + 1) + 3] = np.eye(3)
        a_v[:, 3 * k : 3 * k + 3] = -np.eye(3)
        a_v[:, gi : gi + g_cols] = -dt * (np.eye(3) if basis is None else basis)
        rows += [a_p, a_v]
        rhs += [r_wd[k] @ alpha + 0.5 * dt * dt * g0, r_wd[k] @ beta + dt * g0]
    return np.vstack(rows), np.concatenate(rhs)


def _unobservable_reason(accel_std: float, scale: float, rel_std: float) -> str:
    if accel_std < MIN_ACCEL_STD:
        return f"phone barely accelerated (accel std {accel_std:.2f} < {MIN_ACCEL_STD} m/s^2)"
    if scale <= 0:
        return "solved scale is not positive"
    if rel_std > MAX_SCALE_REL_STD:
        return f"scale too uncertain ({rel_std:.0%} > {MAX_SCALE_REL_STD:.0%})"
    return ""


def align_scale_gravity(
    times: Array,
    centers: Array,
    r_wd: Array,
    gyro: ImuSeries,
    accel: ImuSeries,
    *,
    g_norm: float = GRAVITY,
    refine_iterations: int = 4,
) -> ViAlignment:
    """Solve metric scale, gravity and velocities by linear least squares.

    For consecutive keyframes ``i -> j`` with ``dt = t_j - t_i``::

        s * c_j = s * c_i + v_i * dt + 0.5 * g * dt^2 + R_wd_i @ alpha_ij
        v_j     = v_i + g * dt + R_wd_i @ beta_ij

    The first solve leaves ``g`` free. Refinement then fixes ``|g| = g_norm``
    and solves for a 2-D correction in the plane orthogonal to gravity.

    Args:
        times: ``(N,)`` keyframe times on the camera clock.
        centers: ``(N, 3)`` camera centres from the visual model (unknown scale).
        r_wd: ``(N, 3, 3)`` device-to-world rotations (``R_wc @ R_cd``).
        gyro: Gyroscope samples on the camera clock.
        accel: Accelerometer samples on the camera clock.
        g_norm: Gravity magnitude.
        refine_iterations: Fixed-norm refinement passes.

    Returns:
        The alignment, including whether the scale is trustworthy.
    """
    n = len(times)
    pre = [preintegrate(times[k], times[k + 1], gyro, accel) for k in range(n - 1)]
    in_clip = (accel.t >= times[0]) & (accel.t <= times[-1])
    accel_std = float(np.std(np.linalg.norm(accel.xyz[in_clip], axis=1))) if in_clip.any() else 0.0

    a, b = _build_system(times, centers, r_wd, pre, g_dir=None, g_norm=g_norm)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    g_free = x[3 * n : 3 * n + 3]
    norm = float(np.linalg.norm(g_free))
    g_dir = g_free / norm if norm > _TINY else np.array([0.0, 0.0, -1.0])
    for _ in range(refine_iterations):
        a, b = _build_system(times, centers, r_wd, pre, g_dir=g_dir, g_norm=g_norm)
        x, *_ = np.linalg.lstsq(a, b, rcond=None)
        g_vec = g_norm * g_dir + _tangent_basis(g_dir) @ x[3 * n : 3 * n + 2]
        g_dir = g_vec / np.linalg.norm(g_vec)

    residual = a @ x - b
    sigma2 = float(residual @ residual) / max(len(b) - a.shape[1], 1)
    scale = float(x[-1])
    scale_std = float(np.sqrt(max(sigma2 * np.linalg.pinv(a.T @ a)[-1, -1], 0.0)))
    rel_std = scale_std / abs(scale) if scale != 0 else float("inf")
    reason = _unobservable_reason(accel_std, scale, rel_std)
    return ViAlignment(
        scale=scale,
        gravity=g_norm * g_dir,
        velocities=x[: 3 * n].reshape(n, 3),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        scale_rel_std=rel_std,
        accel_std=accel_std,
        observable=not reason,
        reason=reason,
    )
