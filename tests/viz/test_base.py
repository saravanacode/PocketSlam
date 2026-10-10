from collections.abc import Callable
from pathlib import Path

from pocketslam.core.types import Frame, Pose
from pocketslam.depth.base import ConstantDepthEstimator
from pocketslam.viz.base import TrajectoryWriter, Visualizer


def test_writes_tum_lines_and_skips_lost(tmp_path: Path, make_frame: Callable[..., Frame]) -> None:
    writer = TrajectoryWriter(tmp_path / "s" / "trajectory.tum")
    assert isinstance(writer, Visualizer)
    frame = make_frame()
    depth = ConstantDepthEstimator().estimate(frame)
    writer.show(frame, depth, Pose(timestamp_s=1.0))
    writer.show(frame, depth, None)
    writer.show(frame, depth, Pose(timestamp_s=2.0))
    writer.close()
    lines = writer.path.read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["1.000000", "2.000000"]
