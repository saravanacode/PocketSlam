from collections.abc import Callable

import numpy as np

from pocketslam.core.types import Frame
from pocketslam.depth.base import ConstantDepthEstimator, DepthEstimator


def test_constant_estimator_matches_frame_and_is_relative(make_frame: Callable[..., Frame]) -> None:
    est = ConstantDepthEstimator()
    assert isinstance(est, DepthEstimator)
    depth = est.estimate(make_frame(width=32, height=24, timestamp_s=2.0))
    assert depth.depth.shape == (24, 32)
    assert depth.depth.dtype == np.float32
    assert not depth.is_metric
    assert depth.timestamp_s == 2.0
