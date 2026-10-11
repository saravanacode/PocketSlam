"""Turn per-view depth into one point cloud: scale, gravity-align, back-project, fuse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pocketslam.core.types import MultiViewGeometry


@dataclass(frozen=True, slots=True)
class PointCloud:
    """Coloured points.

    Attributes:
        xyz: ``(M, 3)`` float32 positions in metres (world frame, z up).
        rgb: ``(M, 3)`` uint8 colours.
    """

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8]

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def world_transform(scale: float, rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """4x4 similarity taking model-world points to the output world: ``R @ (s * p)``."""
    t = np.eye(4)
    t[:3, :3] = scale * rotation
    return t


def transform_poses(
    camera_to_world: NDArray[np.float64], scale: float, rotation: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply scale and rotation to ``(N, 4, 4)`` camera-to-world poses.

    Rotations stay orthonormal; only translations are scaled.
    """
    out = camera_to_world.copy()
    out[:, :3, :3] = rotation @ camera_to_world[:, :3, :3]
    out[:, :3, 3] = scale * camera_to_world[:, :3, 3] @ rotation.T
    return out


def backproject(
    depth: NDArray[np.float32], k: NDArray[np.float64], mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Camera-frame 3D points ``(M, 3)`` for pixels where ``mask`` is true."""
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    x = (u + 0.5 - k[0, 2]) / k[0, 0] * z
    y = (v + 0.5 - k[1, 2]) / k[1, 1] * z
    return np.stack([x, y, z], axis=1)


def voxel_downsample(
    xyz: NDArray[np.float64], rgb: NDArray[np.uint8], voxel_m: float
) -> PointCloud:
    """Average all points (and colours) that fall in the same voxel."""
    if len(xyz) == 0:
        return PointCloud(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    keys = np.floor(xyz / voxel_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, xyz)
    colors = np.zeros((len(counts), 3))
    np.add.at(colors, inverse, rgb.astype(np.float64))
    return PointCloud(
        xyz=(sums / counts[:, None]).astype(np.float32),
        rgb=np.clip(colors / counts[:, None], 0, 255).astype(np.uint8),
    )


def fuse(
    geo: MultiViewGeometry,
    scale: float,
    rotation: NDArray[np.float64],
    conf_percentile: float,
    voxel_m: float,
) -> PointCloud:
    """Fuse all views into one point cloud in the output world frame.

    Args:
        geo: Per-view depth and cameras from the model.
        scale: Factor taking model units to metres (1.0 if unknown).
        rotation: Rotation taking the model world to the output world (z up).
        conf_percentile: Per-view confidence percentile below which pixels are dropped.
        voxel_m: Voxel size for downsampling, in output units.
    """
    poses = transform_poses(geo.camera_to_world(), scale, rotation)
    points, colors = [], []
    for i in range(len(geo)):
        conf = geo.confidence[i]
        mask = (conf >= np.percentile(conf, conf_percentile)) & (geo.depth[i] > 0)
        cam = backproject(geo.depth[i], geo.intrinsics[i], mask) * scale
        points.append(cam @ poses[i, :3, :3].T + poses[i, :3, 3])
        colors.append(geo.images[i][mask])
    return voxel_downsample(np.concatenate(points), np.concatenate(colors), voxel_m)
