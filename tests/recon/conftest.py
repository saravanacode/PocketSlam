from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pocketslam.core.types import ImuSample, Intrinsics, SessionMetadata
from pocketslam.record.telemetry import Telemetry
from pocketslam.record.worker import RecordingWorker


@pytest.fixture
def write_recording(tmp_path: Path) -> Callable[..., Path]:
    """Write a small MCAP recording through the real recorder."""

    def _write(frames: int = 12, imu: int = 40, fps: float = 30.0) -> Path:
        path = tmp_path / "rec.mcap"
        worker = RecordingWorker(Telemetry(), path, jpeg_quality=90, max_queue=500)
        k = Intrinsics.from_fov(32, 24, 60.0)
        worker.submit_meta(SessionMetadata(32, 24, "phone"), 100.0)
        for i in range(frames):
            image = np.full((24, 32, 3), i * 10 % 255, dtype=np.uint8)
            worker.submit_frame(image, 100.0 + i / fps, k)
        worker.submit_imu(
            [(ImuSample("gyro", 0.0, 0.1, 0.2, 0.3), 100.0 + j / 100) for j in range(imu)]
            + [(ImuSample("accel", 0.0, 0.0, 9.8, 0.0), 100.0 + j / 100) for j in range(imu)]
        )
        worker.close()
        return path

    return _write
