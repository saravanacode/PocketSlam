from collections.abc import Callable

import numpy as np

from pocketslam.core.types import Frame
from pocketslam.depth.base import ConstantDepthEstimator
from pocketslam.slam.base import SlamBackend, StaticSlam


def test_static_slam_returns_identity(make_frame: Callable[..., Frame]) -> None:
    slam = StaticSlam()
    assert isinstance(slam, SlamBackend)
    frame = make_frame(timestamp_s=3.0)
    pose = slam.track(frame, ConstantDepthEstimator().estimate(frame))
    assert pose is not None
    assert pose.timestamp_s == 3.0
    np.testing.assert_array_equal(pose.matrix, np.eye(4))
