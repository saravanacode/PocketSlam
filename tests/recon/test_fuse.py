import numpy as np
import pytest

from pocketslam.core import so3
from pocketslam.core.types import MultiViewGeometry
from pocketslam.recon.fuse import backproject, fuse, transform_poses, voxel_downsample

K = np.array([[20.0, 0, 8], [0, 20.0, 6], [0, 0, 1]])


def plane_geometry(n: int = 2) -> MultiViewGeometry:
    w2c = np.tile(np.eye(4), (n, 1, 1))
    w2c[1, 0, 3] = -0.5  # second camera 0.5 units to the right
    return MultiViewGeometry(
        depth=np.full((n, 12, 16), 2.0, np.float32),
        confidence=np.ones((n, 12, 16), np.float32),
        intrinsics=np.tile(K, (n, 1, 1)),
        world_to_camera=w2c,
        images=np.full((n, 12, 16, 3), 200, np.uint8),
        is_metric=False,
    )


def test_backproject_center_pixel() -> None:
    mask = np.zeros((12, 16), bool)
    mask[5, 7] = True  # pixel centre (7.5, 5.5) is half a pixel off the principal point
    pts = backproject(np.full((12, 16), 2.0, np.float32), K, mask)
    np.testing.assert_allclose(pts, [[-0.05, -0.05, 2.0]])


def test_transform_poses_scales_translation_only() -> None:
    c2w = np.tile(np.eye(4), (1, 1, 1))
    c2w[0, :3, 3] = [1.0, 0, 0]
    r = so3.exp(np.array([0.0, 0.0, np.pi / 2]))
    out = transform_poses(c2w, 3.0, r)
    np.testing.assert_allclose(out[0, :3, 3], [0, 3.0, 0], atol=1e-12)
    np.testing.assert_allclose(out[0, :3, :3], r)


def test_fuse_puts_plane_at_scaled_depth() -> None:
    cloud = fuse(plane_geometry(), scale=2.0, rotation=np.eye(3), conf_percentile=0, voxel_m=0.05)
    assert len(cloud) > 0
    np.testing.assert_allclose(cloud.xyz[:, 2], 4.0, atol=1e-5)
    assert (cloud.rgb == 200).all()


def test_voxel_downsample_merges_points() -> None:
    xyz = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.5, 0, 0]])
    rgb = np.array([[0, 0, 0], [100, 100, 100], [255, 255, 255]], np.uint8)
    cloud = voxel_downsample(xyz, rgb, 0.01)
    assert len(cloud) == 2
    assert cloud.xyz[:, 0].min() == pytest.approx(0.0015)
    assert len(voxel_downsample(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), 0.01)) == 0
