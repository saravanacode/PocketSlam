"""Marker scale on a synthetic scene: a 16 cm square on a tilted plane, model scale 1/2."""

import numpy as np
import pytest
from pocketslam.calib.marker import (
    edge_lengths,
    estimate_focal,
    estimate_marker_scale,
    lift_corners,
    make_detector,
    rays,
    render_marker,
    triangulate,
)

from pocketslam.core import so3
from pocketslam.core.types import MultiViewGeometry

SIZE_M = 0.16
TRUE_SCALE = 2.0  # model units -> metres
H, W = 240, 320
K = np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]])

# Marker plane in the model world: centred 0.5 units ahead, tilted about x and y.
PLANE_R = so3.exp(np.array([0.3, -0.2, 0.1]))
PLANE_C = np.array([0.02, -0.01, 0.5])
HALF = SIZE_M / TRUE_SCALE / 2
CORNERS_W = (
    np.array([[-HALF, -HALF, 0], [HALF, -HALF, 0], [HALF, HALF, 0], [-HALF, HALF, 0]]) @ PLANE_R.T
    + PLANE_C
)
NORMAL_W = PLANE_R[:, 2]


def camera(offset: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    c2w = np.eye(4)
    c2w[:3, :3] = so3.exp(rotvec)
    c2w[:3, 3] = offset
    return c2w


POSES = [
    camera(np.array([0.0, 0.0, 0.0]), np.zeros(3)),
    camera(np.array([0.08, 0.0, 0.02]), np.array([0.0, -0.12, 0.0])),
    camera(np.array([-0.06, 0.04, 0.0]), np.array([0.06, 0.1, 0.02])),
    camera(np.array([0.0, -0.07, 0.05]), np.array([-0.1, 0.0, 0.0])),
]


def view(c2w: np.ndarray, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Depth map of the infinite marker plane and the marker's corner pixels."""
    r, t = c2w[:3, :3], c2w[:3, 3]
    v, u = np.mgrid[0:H, 0:W]
    d_cam = rays(np.stack([u.ravel() + 0.5, v.ravel() + 0.5], axis=1), K)
    d_world = d_cam @ r.T
    depth = ((PLANE_C - t) @ NORMAL_W / (d_world @ NORMAL_W)).reshape(H, W)
    if noise:
        depth = depth * (1 + noise * np.random.default_rng(0).normal(size=depth.shape))
    cam = (CORNERS_W - t) @ r
    px = np.stack(
        [cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2], cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]],
        axis=1,
    )
    return depth.astype(np.float32), px


def scene(noise: float = 0.0) -> tuple[MultiViewGeometry, dict[int, tuple[int, np.ndarray]]]:
    views = [view(p, noise) for p in POSES]
    n = len(POSES)
    geo = MultiViewGeometry(
        depth=np.stack([d for d, _ in views]),
        confidence=np.ones((n, H, W), np.float32),
        intrinsics=np.tile(K, (n, 1, 1)),
        world_to_camera=np.linalg.inv(np.stack(POSES)),
        images=np.zeros((n, H, W, 3), np.uint8),
        is_metric=False,
    )
    return geo, {i: (18, px) for i, (_, px) in enumerate(views)}


def test_lift_corners_recovers_square() -> None:
    depth, px = view(POSES[1])
    lifted = lift_corners(px, depth, K)
    assert lifted is not None
    np.testing.assert_allclose(edge_lengths(lifted), SIZE_M / TRUE_SCALE, rtol=1e-4)


def test_lift_rejects_non_planar_depth() -> None:
    depth, px = view(POSES[0])
    bumpy = depth * (1 + 0.5 * np.random.default_rng(1).random(depth.shape)).astype(np.float32)
    assert lift_corners(px, bumpy, K) is None
    assert lift_corners(px, np.zeros_like(depth), K) is None


def test_scale_from_views_and_triangulation() -> None:
    geo, detections = scene(noise=0.003)
    result = estimate_marker_scale(detections, geo, SIZE_M)
    assert result is not None
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.01)
    assert result.scale_triangulated == pytest.approx(TRUE_SCALE, rel=0.01)
    assert (result.views_found, result.views_used, result.marker_id) == (4, 4, 18)
    assert result.spread < 0.01
    np.testing.assert_allclose(result.corners_world, CORNERS_W, atol=2e-3)


def test_needs_two_usable_views() -> None:
    geo, detections = scene()
    assert estimate_marker_scale({0: detections[0]}, geo, SIZE_M) is None


def test_rejects_inconsistent_views() -> None:
    geo, detections = scene()
    depth = geo.depth.copy()
    depth[1] *= 2.0  # one view's depth is badly off
    depth[2] *= 0.4
    bad = MultiViewGeometry(
        depth, geo.confidence, geo.intrinsics, geo.world_to_camera, geo.images, False
    )
    assert estimate_marker_scale(detections, bad, SIZE_M) is None


def test_triangulate_intersecting_rays() -> None:
    target = np.array([0.1, -0.2, 1.5])
    origins = np.array([[0.0, 0, 0], [0.5, 0, 0], [0, 0.4, 0.1]])
    np.testing.assert_allclose(triangulate(origins, target - origins), target, atol=1e-9)


def test_detects_rendered_marker_and_filters_id() -> None:
    cv2 = pytest.importorskip("cv2")
    marker = render_marker("DICT_4X4_50", 18, side_px=200)
    photo = np.full((480, 640), 255, np.uint8)
    src = np.float32(
        [[0, 0], [marker.shape[1], 0], [marker.shape[1], marker.shape[0]], [0, marker.shape[0]]]
    )
    dst = np.float32([[180, 90], [470, 120], [450, 400], [160, 370]])
    warp = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(
        marker, warp, (640, 480), dst=photo, borderMode=cv2.BORDER_TRANSPARENT
    )
    rgb = np.stack([photo] * 3, axis=-1)

    found = make_detector("DICT_4X4_50", 18)(rgb)
    assert found is not None
    marker_id, corners = found
    assert marker_id == 18
    pad = (marker.shape[0] - 200) / 2
    square = np.float32([[pad, pad], [pad + 200, pad], [pad + 200, pad + 200], [pad, pad + 200]])
    expected = cv2.perspectiveTransform(square[None], warp)[0]
    np.testing.assert_allclose(corners, expected, atol=1.5)

    assert make_detector("DICT_4X4_50", 7)(rgb) is None
    assert make_detector("DICT_5X5_50", None)(rgb) is None


def test_unknown_dictionary_is_rejected() -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="unknown ArUco dictionary"):
        make_detector("DICT_NOPE", None)


def marker_views(focal: float, tilts_deg: list[float], distance: float = 0.45) -> list[np.ndarray]:
    """Corner pixels of a 16 cm marker seen at 720x1280 from several tilts."""
    half = SIZE_M / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    views = []
    for i, tilt in enumerate(tilts_deg):
        axis = np.array([np.cos(i * 1.3), np.sin(i * 1.3), 0.0])
        r = so3.exp(axis * np.radians(tilt)) @ np.diag([1.0, -1.0, -1.0])  # face the camera
        cam = square @ r.T + np.array([0.02 * np.sin(i), 0.03 * np.cos(i), distance])
        views.append(
            np.stack(
                [cam[:, 0] / cam[:, 2] * focal + 360, cam[:, 1] / cam[:, 2] * focal + 640], axis=1
            )
            + 0.5
        )
    return views


def test_focal_recovered_from_tilted_views() -> None:
    pytest.importorskip("cv2")
    views = marker_views(focal=870.0, tilts_deg=[35, 40, 30, 45, 38, 42, 33, 36, 44, 31])
    est = estimate_focal(views, (720, 1280), SIZE_M)
    assert est is not None
    assert est.reliable, est.reason
    assert est.focal_px == pytest.approx(870.0, rel=0.02)


def test_head_on_views_are_not_trusted() -> None:
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    views = [
        v + rng.normal(scale=0.3, size=(4, 2))
        for v in marker_views(870.0, [1, 2, 1, 2, 1, 2, 1, 2], 1.5)
    ]
    est = estimate_focal(views, (720, 1280), SIZE_M)
    assert est is not None
    assert not est.reliable
    assert "steeper angles" in est.reason


def test_focal_needs_enough_views() -> None:
    pytest.importorskip("cv2")
    assert estimate_focal(marker_views(870.0, [40]), (720, 1280), SIZE_M) is None
    few = estimate_focal(marker_views(870.0, [40, 35, 30]), (720, 1280), SIZE_M)
    assert few is not None
    assert not few.reliable
