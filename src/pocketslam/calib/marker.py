"""Metric scale from a fiducial marker of known size (ArUco).

A square marker with a known edge length, seen in a scan, fixes the scale of
the reconstruction exactly:

1. Detect the marker's four corners in each keyframe (OpenCV ArUco).
2. Per view, fit a plane to the model's depth inside the marker and intersect
   the corner rays with it. That gives the corners in 3D, in model units.
3. ``scale = marker_size_m / mean edge length``; the median over views is used.

Triangulating the corners across views with the model's poses gives a second,
depth-independent estimate used as a cross-check.

Pixel convention: continuous image coordinates with the top-left image corner
at (0, 0), so pixel ``(i, j)`` covers ``[j, j+1) x [i, i+1)``. This matches
``recon.fuse.backproject``. OpenCV reports pixel centres at integer values, so
its corners are shifted by +0.5 here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pocketslam.core.types import MultiViewGeometry

logger = logging.getLogger(__name__)

MIN_SIDE_PX = 24.0
MAX_EDGE_RATIO = 1.25
MAX_PLANE_RESIDUAL = 0.05
MIN_PLANE_POINTS = 20
QUAD_INSET = 0.15
MIN_VIEWS = 2
MAX_SPREAD = 0.20
# Triangulation is only trusted when the camera moved at least this fraction
# of the marker's distance (about 11 degrees of parallax).
MIN_PARALLAX = 0.2
MIN_FOCAL_VIEWS = 6
MAX_FOCAL_VIEWS = 60
MAX_FOCAL_REL_STD = 0.05
_TINY = 1e-9

Array = NDArray[np.float64]
# Returns (marker id, corners) with corners as (4, 2) continuous pixel
# coordinates in the image it was given, ordered around the square.
Detector = Callable[[NDArray[np.uint8]], "tuple[int, Array] | None"]


@dataclass(frozen=True, slots=True)
class MarkerScale:
    """Scale estimate from the marker.

    Attributes:
        marker_id: Id of the detected marker.
        views_found: Keyframes where the marker was detected.
        views_used: Views that passed the quality gates.
        scale: Model units to metres (median over used views).
        spread: Median absolute deviation of the per-view scales over ``scale``.
        scale_triangulated: Scale from corners triangulated across views, or
            None with fewer than two views.
        corners_world: ``(4, 3)`` marker corners in the model world, model units.
    """

    marker_id: int
    views_found: int
    views_used: int
    scale: float
    spread: float
    scale_triangulated: float | None
    corners_world: Array


def make_detector(dictionary: str, marker_id: int | None) -> Detector:
    """Build an ArUco detector for one dictionary, optionally one id.

    Args:
        dictionary: OpenCV name, for example ``"DICT_4X4_50"``.
        marker_id: Only accept this id; None accepts any (the largest wins).

    Raises:
        RuntimeError: If OpenCV is not installed (it comes with the ``recon`` extra).
        ValueError: If the dictionary name is unknown.
    """
    try:
        import cv2  # noqa: PLC0415 - optional extra
    except ImportError as exc:
        msg = "marker detection needs OpenCV. Run: uv sync --extra recon"
        raise RuntimeError(msg) from exc
    if not hasattr(cv2.aruco, dictionary):
        msg = f"unknown ArUco dictionary {dictionary!r}"
        raise ValueError(msg)
    parameters = cv2.aruco.DetectorParameters()
    # Sub-pixel corners: lower distance error at every range (0.1% vs 0.3% at
    # 0.5 m in a blurred, compressed test) without losing far detections.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), parameters
    )

    def detect(image: NDArray[np.uint8]) -> tuple[int, Array] | None:
        corners, ids, _ = detector.detectMarkers(image)
        best: tuple[int, Array] | None = None
        best_side = MIN_SIDE_PX
        for quad, found in zip(corners, [] if ids is None else ids.ravel(), strict=False):
            if marker_id is not None and int(found) != marker_id:
                continue
            pts = np.asarray(quad, dtype=np.float64).reshape(4, 2) + 0.5
            side = float(np.mean(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)))
            if side >= best_side:
                best, best_side = (int(found), pts), side
        return best

    return detect


def render_marker(
    dictionary: str, marker_id: int, side_px: int = 800, margin: float = 0.25
) -> NDArray[np.uint8]:
    """Marker image with the white border ArUco needs around the black square.

    Args:
        dictionary: OpenCV dictionary name.
        marker_id: Marker id within the dictionary.
        side_px: Edge of the black square in pixels.
        margin: White border width as a fraction of ``side_px``.

    Returns:
        ``(H, W)`` uint8 image, white background.
    """
    import cv2  # noqa: PLC0415 - optional extra

    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), marker_id, side_px
    )
    pad = round(side_px * margin)
    return np.asarray(np.pad(marker, pad, constant_values=255), dtype=np.uint8)


def rays(pixels: Array, k: Array) -> Array:
    """Camera-frame ray directions (z = 1) for continuous pixel coordinates."""
    x = (pixels[:, 0] - k[0, 2]) / k[0, 0]
    y = (pixels[:, 1] - k[1, 2]) / k[1, 1]
    return np.stack([x, y, np.ones(len(pixels))], axis=1)


def _inside_quad(quad: Array, height: int, width: int) -> NDArray[np.bool_]:
    """Mask of pixel centres inside a convex quad, shrunk towards its centre."""
    centre = quad.mean(axis=0)
    inner = centre + (quad - centre) * (1.0 - QUAD_INSET)
    v, u = np.mgrid[0:height, 0:width]
    px, py = u + 0.5, v + 0.5
    signs = []
    for a, b in zip(inner, np.roll(inner, -1, axis=0), strict=True):
        signs.append((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]))
    stacked = np.stack(signs)
    return np.asarray((stacked >= 0).all(axis=0) | (stacked <= 0).all(axis=0), dtype=np.bool_)


def lift_corners(corners: Array, depth: NDArray[np.float32], k: Array) -> Array | None:
    """Marker corners in 3D (camera frame) from depth inside the marker.

    Fits a plane to the back-projected depth pixels inside the marker, then
    intersects each corner's ray with it. Corner pixels themselves are not
    read: depth is least reliable exactly at edges.

    Args:
        corners: ``(4, 2)`` corner pixels at the depth map's resolution.
        depth: ``(H, W)`` depth along the camera z axis.
        k: 3x3 intrinsics at the depth map's resolution.

    Returns:
        ``(4, 3)`` corners in the units of ``depth``, or None if the marker
        covers too few valid pixels or is not planar in the depth map.
    """
    mask = _inside_quad(corners, *depth.shape) & (depth > 0)
    if int(mask.sum()) < MIN_PLANE_POINTS:
        return None
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    points = rays(np.stack([u + 0.5, v + 0.5], axis=1), k) * z[:, None]
    centroid = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    residual = singular[2] / np.sqrt(len(points))
    if residual > MAX_PLANE_RESIDUAL * float(np.mean(z)):
        return None
    directions = rays(corners, k)
    denom = directions @ normal
    if np.any(np.abs(denom) < _TINY):
        return None
    t = float(centroid @ normal) / denom
    if np.any(t <= 0):
        return None
    return np.asarray(directions * t[:, None], dtype=np.float64)


def edge_lengths(corners_3d: Array) -> Array:
    """The four edge lengths of a quad given as ``(4, 3)`` corners in order."""
    return np.asarray(
        np.linalg.norm(corners_3d - np.roll(corners_3d, -1, axis=0), axis=1), dtype=np.float64
    )


def triangulate(origins: Array, directions: Array) -> Array:
    """Point closest to a set of rays (least squares).

    Args:
        origins: ``(N, 3)`` ray origins.
        directions: ``(N, 3)`` ray directions (any length).
    """
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projectors = np.eye(3) - d[:, :, None] * d[:, None, :]
    lhs = projectors.sum(axis=0)
    rhs = np.einsum("nij,nj->i", projectors, origins)
    return np.asarray(np.linalg.lstsq(lhs, rhs, rcond=None)[0], dtype=np.float64)


def estimate_marker_scale(
    detections: dict[int, tuple[int, Array]],
    geo: MultiViewGeometry,
    size_m: float,
) -> MarkerScale | None:
    """Combine per-view marker detections into one scale.

    Args:
        detections: Keyframe index to ``(marker id, corners)`` with corners at
            the model's resolution.
        geo: The model's depth, intrinsics and poses.
        size_m: Edge of the marker's black square in metres.

    Returns:
        The scale, or None when too few views pass the quality gates or the
        views disagree by more than ``MAX_SPREAD``.
    """
    c2w = geo.camera_to_world()
    scales: list[float] = []
    world: list[Array] = []
    used: list[int] = []
    for index, (_, corners) in sorted(detections.items()):
        lifted = lift_corners(corners, geo.depth[index], geo.intrinsics[index])
        if lifted is None:
            continue
        edges = edge_lengths(lifted)
        if edges.min() <= 0 or edges.max() / edges.min() > MAX_EDGE_RATIO:
            continue
        scales.append(size_m / float(edges.mean()))
        world.append(lifted @ c2w[index, :3, :3].T + c2w[index, :3, 3])
        used.append(index)
    if len(scales) < MIN_VIEWS:
        logger.warning(
            "marker seen but not usable",
            extra={"views_found": len(detections), "views_used": len(scales)},
        )
        return None
    scale = float(np.median(scales))
    spread = float(np.median(np.abs(np.array(scales) - scale)) / scale)
    if spread > MAX_SPREAD:
        logger.warning("marker scale inconsistent across views", extra={"spread": spread})
        return None

    origins = c2w[used, :3, 3]
    tri = np.stack(
        [
            triangulate(
                origins,
                np.stack(
                    [
                        rays(detections[i][1][j : j + 1], geo.intrinsics[i])[0] @ c2w[i, :3, :3].T
                        for i in used
                    ]
                ),
            )
            for j in range(4)
        ]
    )
    tri_edges = edge_lengths(tri)
    baseline = float(np.linalg.norm(origins.max(axis=0) - origins.min(axis=0)))
    distance = float(np.median(np.linalg.norm(origins - tri.mean(axis=0), axis=1)))
    # Triangulation needs the camera to have moved; with little parallax the
    # result is noise, so it is not reported.
    tri_ok = baseline > MIN_PARALLAX * distance and tri_edges.min() > 0
    marker_id = next(iter(detections.values()))[0]
    return MarkerScale(
        marker_id=marker_id,
        views_found=len(detections),
        views_used=len(scales),
        scale=scale,
        spread=spread,
        scale_triangulated=size_m / float(tri_edges.mean()) if tri_ok else None,
        corners_world=np.asarray(np.median(np.stack(world), axis=0), dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class FocalEstimate:
    """Focal length measured from marker views.

    Attributes:
        focal_px: Focal length in pixels at the resolution of the detections.
        rel_std: Its standard deviation divided by the value.
        views: Marker views used.
        rms_px: Reprojection error.
        reliable: True when enough views constrain the value well.
        reason: Why it is not reliable, empty when it is.
    """

    focal_px: float
    rel_std: float
    views: int
    rms_px: float
    reliable: bool
    reason: str


def estimate_focal(
    corner_sets: list[Array], image_size: tuple[int, int], size_m: float
) -> FocalEstimate | None:
    """Measure the camera's focal length from many views of the marker.

    Solves one focal length (square pixels, principal point at the image
    centre, no lens distortion) with OpenCV's calibration. A flat marker seen
    head-on says almost nothing about focal length; the views must show it at
    steep angles and large in the image, which the reported uncertainty reflects.

    Args:
        corner_sets: ``(4, 2)`` corners per view, in this module's pixel convention.
        image_size: ``(width, height)`` of the images the corners came from.
        size_m: Edge of the marker's black square in metres.

    Returns:
        The estimate, or None with fewer than two views.
    """
    import cv2  # noqa: PLC0415 - optional extra

    if len(corner_sets) < MIN_VIEWS:
        return None
    step = max(1, len(corner_sets) // MAX_FOCAL_VIEWS)
    views = corner_sets[::step]
    half = size_m / 2
    square = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32
    )
    width, height = image_size
    guess = np.array(
        [[max(image_size), 0, width / 2], [0, max(image_size), height / 2], [0, 0, 1]], np.float64
    )
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, k, _, _, _, std_intrinsics, _, _ = cv2.calibrateCameraExtended(
        [square] * len(views),
        [(c - 0.5).astype(np.float32) for c in views],
        (width, height),
        guess,
        np.zeros(5),
        flags=flags,
    )
    focal = float(k[0, 0])
    # With the aspect ratio fixed, OpenCV reports the uncertainty under fy and 0 under fx.
    rel_std = float(np.max(std_intrinsics.ravel()[:2])) / focal
    reason = ""
    if len(views) < MIN_FOCAL_VIEWS:
        reason = f"only {len(views)} marker views; at least {MIN_FOCAL_VIEWS} are needed"
    elif rel_std > MAX_FOCAL_REL_STD:
        reason = (
            f"focal length is uncertain by {rel_std:.0%} (limit {MAX_FOCAL_REL_STD:.0%}): "
            "show the marker larger and at steeper angles"
        )
    return FocalEstimate(focal, rel_std, len(views), float(rms), not reason, reason)
