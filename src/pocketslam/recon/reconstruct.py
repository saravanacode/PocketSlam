"""Offline reconstruction of one recording: depth + poses + IMU scale -> fused 3D.

Steps (see RULES.md "Offline reconstruction"):

1. Pick keyframes (about ``keyframe_fps``, at most ``max_frames``).
2. Depth Anything 3 any-view: per-frame depth, camera poses, intrinsics (no scale).
3. Camera-IMU time offset and phone orientation from gyro vs visual rotation.
4. Visual-inertial least squares: metric scale and gravity.
5. Optional metric-depth cross-check; it replaces the IMU scale when that is
   not observable.
6. Optional fiducial marker of known size: its scale wins over every other
   source, and the correction it implies for the metric-depth scale is
   remembered for scans without a marker.
7. Gravity-align (z up), scale and fuse into one point cloud.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pocketslam.calib.marker import Detector, MarkerScale, estimate_marker_scale
from pocketslam.calib.store import ScaleCalibration, device_label
from pocketslam.config import ReconSettings
from pocketslam.core import so3
from pocketslam.core.types import MultiViewGeometry
from pocketslam.recon.fuse import PointCloud, fuse, transform_poses
from pocketslam.recon.reader import Recording
from pocketslam.slam.vi_align import (
    R_CD_PORTRAIT,
    ViAlignment,
    align_scale_gravity,
    estimate_camera_from_device,
    estimate_time_offset,
    gravity_from_accel,
    samples_in_clip,
)

logger = logging.getLogger(__name__)

MIN_IMU_SAMPLES = 20
MIN_OFFSET_CORRELATION = 0.5
SCALE_DISAGREEMENT_WARN = 0.15
UP = np.array([0.0, 0.0, 1.0])
_MIN_DEPTH = 1e-6


class MultiViewEstimator(Protocol):
    """Joint depth and cameras for a set of frames (see ``depth.da3.DA3MultiView``)."""

    def estimate_views(self, images: Sequence[NDArray[np.uint8]]) -> MultiViewGeometry:
        """Joint depth and cameras for ``images``."""
        ...


class MetricEstimator(Protocol):
    """Metric monocular depth (see ``depth.da3.DA3Metric``)."""

    def metric_depth(
        self, images: Sequence[NDArray[np.uint8]], focal_px: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        """Depth in metres for ``images``."""
        ...


@dataclass
class ReconResult:
    """Everything the exporters need.

    Attributes:
        geometry: Raw model output for the keyframes (model units, model world).
        times: ``(N,)`` keyframe times.
        frame_indices: Keyframe indices into the recording.
        scale: Model units to metres (1.0 when no scale source worked).
        scale_source: ``"marker"``, ``"imu"``, ``"metric-depth (calibrated)"``,
            ``"metric-depth"`` or ``"none"``.
        rotation: Model world to output world (z up).
        camera_to_world: ``(N, 4, 4)`` poses in the output world, metres.
        cloud: Fused point cloud in the output world.
        summary: JSON-ready diagnostics.
    """

    geometry: MultiViewGeometry
    times: NDArray[np.float64]
    frame_indices: NDArray[np.int64]
    scale: float
    scale_source: str
    rotation: NDArray[np.float64]
    camera_to_world: NDArray[np.float64]
    cloud: PointCloud
    summary: dict[str, Any] = field(default_factory=dict)


def select_keyframes(times: NDArray[np.float64], fps: float, max_frames: int) -> NDArray[np.int64]:
    """Indices of frames spaced at least ``1/fps`` apart, then evenly thinned to ``max_frames``."""
    picked = [0]
    for i in range(1, len(times)):
        if times[i] - times[picked[-1]] >= 1.0 / fps - 1e-6:
            picked.append(i)
    idx = np.array(picked, dtype=np.int64)
    if len(idx) > max_frames:
        idx = idx[np.round(np.linspace(0, len(idx) - 1, max_frames)).astype(np.int64)]
    return idx


def metric_scale(
    geo: MultiViewGeometry, metric_depth: NDArray[np.float32], conf_percentile: float
) -> float:
    """Median ratio of metric depth to model depth over confident pixels."""
    ratios = []
    for i in range(len(geo)):
        conf = geo.confidence[i]
        mask = (conf >= np.percentile(conf, conf_percentile)) & (geo.depth[i] > _MIN_DEPTH)
        mask &= metric_depth[i] > _MIN_DEPTH
        ratios.append(metric_depth[i][mask] / geo.depth[i][mask])
    return float(np.median(np.concatenate(ratios)))


def _portrait_prior(meta: dict[str, Any] | None) -> NDArray[np.float64]:
    angle = float((meta or {}).get("orientation", 0) or 0)
    return np.asarray(
        so3.exp(np.array([0.0, 0.0, np.radians(angle)])) @ R_CD_PORTRAIT, dtype=np.float64
    )


def _imu_alignment(
    recording: Recording, times: NDArray[np.float64], c2w: NDArray[np.float64]
) -> tuple[ViAlignment | None, NDArray[np.float64] | None, dict[str, Any]]:
    """Time offset, phone orientation, then scale and gravity from the IMU."""
    if min(len(recording.gyro.t), len(recording.accel.t)) < MIN_IMU_SAMPLES:
        return None, None, {"observable": False, "reason": "recording has no IMU samples"}
    overlap = min(samples_in_clip(times, recording.gyro), samples_in_clip(times, recording.accel))
    if overlap < MIN_IMU_SAMPLES:
        # Seen when the video stream lagged seconds behind the IMU while recording.
        lag = float(recording.accel.t[0] - times[0])
        reason = (
            f"IMU samples do not overlap the frames in time (IMU starts {lag:+.1f} s "
            "from the first frame); the video was lagging while this clip was recorded"
        )
        logger.warning(reason)
        return None, None, {"observable": False, "reason": reason}
    r_wc, centers = c2w[:, :3, :3], c2w[:, :3, 3]
    offset, corr = estimate_time_offset(times, r_wc, recording.gyro)
    if corr < MIN_OFFSET_CORRELATION:
        logger.warning("weak gyro/visual match; keeping zero time offset", extra={"corr": corr})
        offset = 0.0
    gyro = recording.gyro.shifted(offset)
    accel = recording.accel.shifted(offset)
    r_cd, rot_err = estimate_camera_from_device(
        times, r_wc, gyro, prior=_portrait_prior(recording.meta)
    )
    r_wd = r_wc @ r_cd
    vi = align_scale_gravity(times, centers, r_wd, gyro, accel)
    gravity_dir = (
        vi.gravity / np.linalg.norm(vi.gravity)
        if vi.observable
        else gravity_from_accel(times, r_wd, accel)
    )
    info = {
        "time_offset_ms": round(offset * 1000, 1),
        "offset_correlation": round(corr, 3),
        "camera_from_device": np.round(r_cd, 3).tolist(),
        "orientation_fit_rad_s": None if np.isnan(rot_err) else round(rot_err, 4),
        "scale": round(vi.scale, 5),
        "scale_rel_std": round(vi.scale_rel_std, 4),
        "accel_std": round(vi.accel_std, 3),
        "residual_rms": round(vi.residual_rms, 5),
        "observable": vi.observable,
        "reason": vi.reason,
    }
    return vi, gravity_dir, info


def _find_marker(
    images: Sequence[NDArray[np.uint8]],
    geo: MultiViewGeometry,
    settings: ReconSettings,
    detect: Detector | None,
) -> MarkerScale | None:
    """Detect the marker in the full-resolution keyframes and estimate the scale."""
    if detect is None or settings.marker_size_m is None:
        return None
    height, width = geo.depth.shape[1:]
    detections = {}
    for i, image in enumerate(images):
        found = detect(image)
        if found is not None:
            marker_id, corners = found
            to_model = np.array([width / image.shape[1], height / image.shape[0]])
            detections[i] = (marker_id, corners * to_model)
    if not detections:
        logger.info("no marker found in this clip")
        return None
    return estimate_marker_scale(detections, geo, settings.marker_size_m)


def _apply_calibrated_focal(
    geo: MultiViewGeometry, focal_ratio: float
) -> tuple[MultiViewGeometry, dict[str, Any]]:
    """Replace the model's focal length with the measured one (square pixels)."""
    height, width = geo.depth.shape[1:]
    focal = focal_ratio * max(height, width)
    k = geo.intrinsics.copy()
    estimated = float(np.median(k[:, [0, 1], [0, 1]]))
    k[:, 0, 0] = k[:, 1, 1] = focal
    info = {
        "model_estimate_px": round(estimated, 1),
        "calibrated_px": round(focal, 1),
        "ratio": round(focal / estimated, 3),
        "at_resolution": [height, width],
    }
    return replace(geo, intrinsics=k), info


def _choose_scale(
    marker: MarkerScale | None,
    vi: ViAlignment | None,
    scale_metric: float | None,
    correction: tuple[float, int, bool] | None,
    summary: dict[str, Any],
) -> tuple[float, str]:
    """Pick the scale: marker, then IMU, then metric depth (calibrated if possible)."""
    if marker is not None:
        return marker.scale, "marker"
    if vi is not None and vi.observable:
        if scale_metric is not None:
            disagreement = abs(vi.scale - scale_metric) / scale_metric
            summary["scale_disagreement"] = round(disagreement, 3)
            if disagreement > SCALE_DISAGREEMENT_WARN:
                logger.warning(
                    "IMU and metric-depth scales disagree",
                    extra={"imu": vi.scale, "metric": scale_metric, "ratio": disagreement},
                )
        return vi.scale, "imu"
    if scale_metric is not None and correction is not None:
        factor, samples, same_device = correction
        summary["calibration"] = {
            "factor": round(factor, 4),
            "samples": samples,
            "same_phone": same_device,
        }
        return scale_metric * factor, "metric-depth (calibrated)"
    if scale_metric is not None:
        return scale_metric, "metric-depth"
    logger.warning("no metric scale available; output is in model units")
    return 1.0, "none"


def reconstruct(
    recording: Recording,
    multiview: MultiViewEstimator,
    metric: MetricEstimator | None,
    settings: ReconSettings,
    *,
    detect: Detector | None = None,
    calibration: ScaleCalibration | None = None,
) -> ReconResult:
    """Reconstruct ``recording``; see the module docstring for the steps.

    Args:
        recording: Frames, IMU and metadata of one clip.
        multiview: Joint depth and pose model.
        metric: Metric depth model for the scale check, or None.
        settings: Reconstruction settings.
        detect: Marker detector (``recon.marker.make_detector``); used when
            ``settings.marker_size_m`` is set.
        calibration: Store for the remembered marker correction, or None.
    """
    started = time.perf_counter()
    idx = select_keyframes(recording.frame_times, settings.keyframe_fps, settings.max_frames)
    times = recording.frame_times[idx]
    images = [recording.decode(int(i)) for i in idx]
    logger.info("running multi-view depth", extra={"frames": len(idx)})
    t0 = time.perf_counter()
    geo = multiview.estimate_views(images)
    model_s = time.perf_counter() - t0
    c2w = geo.camera_to_world()
    device = device_label(recording.meta)
    focal_info: dict[str, Any] | None = None
    if calibration is not None and settings.use_calibrated_intrinsics:
        measured = calibration.focal_ratio(device)
        if measured is not None:
            geo, focal_info = _apply_calibrated_focal(geo, measured[0])
            focal_info["samples"] = measured[1]

    vi, gravity_dir, imu_info = _imu_alignment(recording, times, c2w)
    summary: dict[str, Any] = {
        "recording": str(recording.path),
        "recording_truncated": recording.truncated,
        "frames_in_recording": len(recording),
        "keyframes": len(idx),
        "clip_seconds": round(float(times[-1] - times[0]), 3),
        "model_seconds": round(model_s, 2),
        "model_resolution": list(geo.depth.shape[1:]),
        "imu_samples": {"gyro": len(recording.gyro.t), "accel": len(recording.accel.t)},
        "imu": imu_info,
        "focal": focal_info,
    }

    scale_metric: float | None = None
    if metric is not None:
        focal = geo.intrinsics[:, [0, 1], [0, 1]].mean(axis=1)
        t0 = time.perf_counter()
        depth_m = metric.metric_depth(images, focal)
        scale_metric = metric_scale(geo, depth_m, settings.conf_percentile)
        summary["metric_depth"] = {
            "scale": round(scale_metric, 5),
            "seconds": round(time.perf_counter() - t0, 2),
        }

    marker = _find_marker(images, geo, settings, detect)
    correction = None
    if calibration is not None and settings.use_calibration:
        # The marker scale is only as good as the focal length. Remember its
        # correction only when that has been measured for this phone.
        if marker is not None and scale_metric is not None and focal_info is not None:
            calibration.add_ratio(device, marker.scale / scale_metric, recording.path.stem)
        elif marker is None:
            correction = calibration.ratio(device)
    scale, source = _choose_scale(marker, vi, scale_metric, correction, summary)

    rotation = (
        so3.between(-gravity_dir, UP) if gravity_dir is not None else np.eye(3, dtype=np.float64)
    )
    if marker is not None:
        tri = marker.scale_triangulated
        summary["marker"] = {
            "id": marker.marker_id,
            "size_m": settings.marker_size_m,
            "views_found": marker.views_found,
            "views_used": marker.views_used,
            "scale": round(marker.scale, 5),
            "spread": round(marker.spread, 4),
            "scale_triangulated": None if tri is None else round(tri, 5),
            "metric_depth_ratio": (
                None if scale_metric is None else round(marker.scale / scale_metric, 4)
            ),
            "corners_world": np.round((scale * marker.corners_world) @ rotation.T, 4).tolist(),
            "focal_calibrated": focal_info is not None,
        }
        if focal_info is None:
            logger.warning(
                "marker scale used with the model's own focal length estimate; "
                "run pocketslam-calibrate for an exact scale"
            )
    elif settings.marker_size_m is not None:
        summary["marker"] = None
    cloud = fuse(geo, scale, rotation, settings.conf_percentile, settings.voxel_m)
    poses = transform_poses(c2w, scale, rotation)
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    summary.update(
        {
            "scale": round(scale, 5),
            "scale_source": source,
            "gravity_aligned": gravity_dir is not None,
            "points": len(cloud),
            "path_length_m": round(float(np.sum(steps)), 3),
            "total_seconds": round(time.perf_counter() - started, 2),
        }
    )
    return ReconResult(
        geometry=geo,
        times=times,
        frame_indices=idx,
        scale=scale,
        scale_source=source,
        rotation=rotation,
        camera_to_world=poses,
        cloud=cloud,
        summary=summary,
    )
