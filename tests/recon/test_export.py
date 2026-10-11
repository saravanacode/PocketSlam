from collections import Counter
from pathlib import Path

import numpy as np
from mcap.reader import make_reader

from pocketslam.core.topics import (
    TOPIC_CALIBRATION,
    TOPIC_IMAGE,
    TOPIC_RECON_DEPTH,
    TOPIC_RECON_POINTS,
    TOPIC_RECON_TRAJECTORY,
    TOPIC_TF,
)
from pocketslam.core.types import MultiViewGeometry
from pocketslam.recon.export import colorize_depth, export, turbo
from pocketslam.recon.fuse import PointCloud
from pocketslam.recon.reconstruct import ReconResult


def make_result(n: int = 3) -> ReconResult:
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 0, 3] = np.arange(n) * 0.1
    geo = MultiViewGeometry(
        depth=np.linspace(1, 3, n * 8 * 10, dtype=np.float32).reshape(n, 8, 10),
        confidence=np.ones((n, 8, 10), np.float32),
        intrinsics=np.tile(np.array([[10.0, 0, 5], [0, 10.0, 4], [0, 0, 1]]), (n, 1, 1)),
        world_to_camera=np.linalg.inv(c2w),
        images=np.zeros((n, 8, 10, 3), np.uint8),
        is_metric=False,
    )
    cloud = PointCloud(
        np.array([[0, 0, 1], [1, 0, 1]], np.float32), np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    )
    return ReconResult(
        geometry=geo,
        times=100.0 + np.arange(n) * 0.1,
        frame_indices=np.arange(n),
        scale=2.0,
        scale_source="imu",
        rotation=np.eye(3),
        camera_to_world=c2w,
        cloud=cloud,
        summary={"scale": 2.0},
    )


def test_writes_all_outputs(tmp_path: Path) -> None:
    paths = export(make_result(), tmp_path / "out")
    with paths["mcap"].open("rb") as f:
        topics = Counter(ch.topic for _, ch, _ in make_reader(f).iter_messages())
    assert topics[TOPIC_RECON_POINTS] == topics[TOPIC_RECON_TRAJECTORY] == 1
    assert topics[TOPIC_TF] == topics[TOPIC_IMAGE] == topics[TOPIC_RECON_DEPTH] == 3
    assert topics[TOPIC_CALIBRATION] == 3

    ply = paths["ply"].read_bytes()
    header, body = ply.split(b"end_header\n")
    assert b"element vertex 2" in header
    assert len(body) == 2 * (3 * 4 + 3)

    lines = paths["trajectory"].read_text().splitlines()
    assert len(lines) == 3
    assert lines[1].split()[:2] == ["100.100000", "0.100000"]

    depth = np.load(paths["depth"])
    assert depth["depth_m"].max() == np.float32(6.0)  # scaled by 2
    assert '"scale": 2.0' in paths["summary"].read_text()


def test_colour_maps() -> None:
    lut = turbo(np.array([0.0, 0.5, 1.0]))
    assert lut.shape == (3, 3)
    assert lut.dtype == np.uint8
    near = colorize_depth(np.array([[1.0]], np.float32), 1.0, 3.0)[0, 0]
    far = colorize_depth(np.array([[2.6]], np.float32), 1.0, 3.0)[0, 0]
    assert near[0] > near[2]  # near is red
    assert far[2] > far[0]  # far is blue (the very far end of turbo is dark purple)
