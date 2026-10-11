"""Write a reconstruction to disk: Foxglove MCAP, PLY, TUM trajectory, depth, summary."""

from __future__ import annotations

import logging
from pathlib import Path

import foxglove
import numpy as np
from foxglove.channels import (
    CameraCalibrationChannel,
    CompressedImageChannel,
    FrameTransformsChannel,
    PointCloudChannel,
    PosesInFrameChannel,
)
from foxglove.messages import (
    CameraCalibration,
    CompressedImage,
    FrameTransform,
    FrameTransforms,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    PosesInFrame,
    Quaternion,
    Timestamp,
    Vector3,
)
from numpy.typing import NDArray

from pocketslam.core.jsonsafe import dumps
from pocketslam.core.topics import (
    TOPIC_CALIBRATION,
    TOPIC_IMAGE,
    TOPIC_RECON_DEPTH,
    TOPIC_RECON_POINTS,
    TOPIC_RECON_TRAJECTORY,
    TOPIC_TF,
)
from pocketslam.core.types import Pose as TumPose
from pocketslam.core.types import rotation_to_quaternion
from pocketslam.recon.fuse import PointCloud as Cloud
from pocketslam.recon.reconstruct import ReconResult
from pocketslam.record.worker import encode_jpeg

logger = logging.getLogger(__name__)

WORLD = "world"
CAMERA = "camera"
JPEG_QUALITY = 90


def _stamp(t_s: float) -> Timestamp:
    ns = round(t_s * 1e9)
    return Timestamp(ns // 1_000_000_000, ns % 1_000_000_000)


def _vec_quat(m: NDArray[np.float64]) -> tuple[Vector3, Quaternion]:
    x, y, z, w = rotation_to_quaternion(m[:3, :3])
    t = m[:3, 3]
    return Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2])), Quaternion(x=x, y=y, z=z, w=w)


def _pose(m: NDArray[np.float64]) -> Pose:
    position, orientation = _vec_quat(m)
    return Pose(position=position, orientation=orientation)


def turbo(x: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Turbo colour map (polynomial fit) for values in [0, 1]; returns ``(..., 3)`` uint8."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (
        4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))
    )
    g = 0.09140261 + x * (
        2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))
    )
    b = 0.1066733 + x * (
        12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))
    )
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)


def colorize_depth(depth: NDArray[np.float32], near: float, far: float) -> NDArray[np.uint8]:
    """Near is warm (red), far is cool (blue)."""
    scaled = (far - depth.astype(np.float64)) / max(far - near, 1e-9)
    return turbo(scaled)


def write_ply(path: Path, cloud: Cloud) -> None:
    """Binary little-endian PLY with float xyz and uchar rgb."""
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(cloud)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rows = np.empty(len(cloud), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rows["xyz"] = cloud.xyz
    rows["rgb"] = cloud.rgb
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.tobytes())


def write_tum(path: Path, times: NDArray[np.float64], camera_to_world: NDArray[np.float64]) -> None:
    """Camera trajectory in TUM format (one line per keyframe)."""
    lines = [TumPose(float(t), m).to_tum() for t, m in zip(times, camera_to_world, strict=True)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cloud_message(cloud: Cloud, t_s: float) -> PointCloud:
    rows = np.empty(len(cloud), dtype=[("xyz", "<f4", 3), ("rgba", "u1", 4)])
    rows["xyz"] = cloud.xyz
    rows["rgba"][:, :3] = cloud.rgb
    rows["rgba"][:, 3] = 255
    f32, u8 = PackedElementFieldNumericType.Float32, PackedElementFieldNumericType.Uint8
    return PointCloud(
        timestamp=_stamp(t_s),
        frame_id=WORLD,
        pose=Pose(position=Vector3(), orientation=Quaternion(w=1.0)),
        point_stride=16,
        fields=[
            PackedElementField(name="x", offset=0, type=f32),
            PackedElementField(name="y", offset=4, type=f32),
            PackedElementField(name="z", offset=8, type=f32),
            PackedElementField(name="red", offset=12, type=u8),
            PackedElementField(name="green", offset=13, type=u8),
            PackedElementField(name="blue", offset=14, type=u8),
            PackedElementField(name="alpha", offset=15, type=u8),
        ],
        data=rows.tobytes(),
    )


def write_mcap(path: Path, result: ReconResult) -> None:
    """Reconstruction for the Foxglove 3D panel (fixed frame ``world``, z up)."""
    ctx = foxglove.Context()
    points = PointCloudChannel(TOPIC_RECON_POINTS, context=ctx)
    tf = FrameTransformsChannel(TOPIC_TF, context=ctx)
    trajectory = PosesInFrameChannel(TOPIC_RECON_TRAJECTORY, context=ctx)
    images = CompressedImageChannel(TOPIC_IMAGE, context=ctx)
    depth = CompressedImageChannel(TOPIC_RECON_DEPTH, context=ctx)
    calibration = CameraCalibrationChannel(TOPIC_CALIBRATION, context=ctx)

    geo = result.geometry
    valid = geo.depth[geo.depth > 0]
    near, far = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
    t0 = float(result.times[0])
    with foxglove.open_mcap(path, allow_overwrite=True, context=ctx):
        points.log(_cloud_message(result.cloud, t0), log_time=round(t0 * 1e9))
        trajectory.log(
            PosesInFrame(
                timestamp=_stamp(t0),
                frame_id=WORLD,
                poses=[_pose(m) for m in result.camera_to_world],
            ),
            log_time=round(t0 * 1e9),
        )
        for i, t in enumerate(result.times):
            ns = round(float(t) * 1e9)
            stamp = _stamp(float(t))
            translation, rotation = _vec_quat(result.camera_to_world[i])
            tf.log(
                FrameTransforms(
                    transforms=[
                        FrameTransform(
                            timestamp=stamp,
                            parent_frame_id=WORLD,
                            child_frame_id=CAMERA,
                            translation=translation,
                            rotation=rotation,
                        )
                    ]
                ),
                log_time=ns,
            )
            k = geo.intrinsics[i]
            h, w = geo.depth.shape[1:]
            calibration.log(
                CameraCalibration(
                    timestamp=stamp,
                    frame_id=CAMERA,
                    width=w,
                    height=h,
                    distortion_model="plumb_bob",
                    D=[0.0] * 5,
                    K=k.reshape(-1).tolist(),
                    R=np.eye(3).reshape(-1).tolist(),
                    P=[k[0, 0], 0.0, k[0, 2], 0.0, 0.0, k[1, 1], k[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0],
                ),
                log_time=ns,
            )
            images.log(
                CompressedImage(
                    timestamp=stamp,
                    frame_id=CAMERA,
                    data=encode_jpeg(geo.images[i], JPEG_QUALITY),
                    format="jpeg",
                ),
                log_time=ns,
            )
            depth.log(
                CompressedImage(
                    timestamp=stamp,
                    frame_id=CAMERA,
                    data=encode_jpeg(
                        colorize_depth(geo.depth[i], float(near), float(far)), JPEG_QUALITY
                    ),
                    format="jpeg",
                ),
                log_time=ns,
            )


def export(result: ReconResult, out_dir: Path) -> dict[str, Path]:
    """Write all outputs to ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mcap": out_dir / "recon.mcap",
        "ply": out_dir / "points.ply",
        "trajectory": out_dir / "trajectory.tum",
        "depth": out_dir / "depth.npz",
        "summary": out_dir / "summary.json",
    }
    write_mcap(paths["mcap"], result)
    write_ply(paths["ply"], result.cloud)
    write_tum(paths["trajectory"], result.times, result.camera_to_world)
    geo = result.geometry
    np.savez_compressed(
        paths["depth"],
        depth_m=(geo.depth * result.scale).astype(np.float32),
        confidence=geo.confidence,
        intrinsics=geo.intrinsics,
        camera_to_world=result.camera_to_world,
        times=result.times,
    )
    paths["summary"].write_text(dumps(result.summary, indent=2), encoding="utf-8")
    logger.info("reconstruction written", extra={"dir": str(out_dir)})
    return paths
