import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pocketslam.core.types import ImuSample, Intrinsics, SessionMetadata
from pocketslam.record.telemetry import (
    TOPIC_ACCEL,
    TOPIC_CALIBRATION,
    TOPIC_GYRO,
    TOPIC_IMAGE,
    TOPIC_META,
    Telemetry,
)
from pocketslam.record.worker import RecordingWorker, encode_jpeg

from .conftest import read_topics


def test_encode_jpeg_round_trips_size() -> None:
    data = encode_jpeg(np.zeros((48, 64, 3), dtype=np.uint8), 80)
    assert Image.open(io.BytesIO(data)).size == (64, 48)


def test_records_frames_imu_and_meta(tmp_path: Path) -> None:
    path = tmp_path / "s.mcap"
    worker = RecordingWorker(Telemetry(), path, jpeg_quality=80, max_queue=100)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    k = Intrinsics.from_fov(64, 48, 60.0)
    worker.submit_meta(SessionMetadata(64, 48, "phone"), 1.0)
    for i in range(3):
        worker.submit_frame(image, 1.0 + i / 30, k)
    worker.submit_imu(
        [(ImuSample("gyro", 0.0, 0, 0, 1), 1.0), (ImuSample("accel", 0.0, 0, 9.8, 0), 1.0)]
    )
    worker.submit_imu([])
    worker.close()

    assert (worker.frames_written, worker.imu_written, worker.dropped) == (3, 2, 0)
    topics = read_topics(path)
    assert topics[TOPIC_IMAGE] == 3
    assert topics[TOPIC_CALIBRATION] == 1  # logged again only when intrinsics change
    assert topics[TOPIC_GYRO] == topics[TOPIC_ACCEL] == topics[TOPIC_META] == 1


def test_live_only_without_path() -> None:
    worker = RecordingWorker(Telemetry(), None, jpeg_quality=80, max_queue=10)
    worker.submit_frame(np.zeros((8, 8, 3), dtype=np.uint8), 0.0, Intrinsics.from_fov(8, 8, 60.0))
    worker.close()
    assert worker.frames_written == 1


def test_clip_contains_only_items_between_start_and_stop(tmp_path: Path) -> None:
    worker = RecordingWorker(Telemetry(), None, jpeg_quality=80, max_queue=100)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    k = Intrinsics.from_fov(16, 16, 60.0)
    worker.submit_meta(SessionMetadata(16, 16, "phone"), 0.5)
    worker.submit_frame(image, 1.0, k)  # before the clip
    clip_path = tmp_path / "clip.mcap"
    assert worker.start_clip(clip_path).result(timeout=5) is None
    for i in range(4):
        worker.submit_frame(image, 2.0 + i * 0.1, k)
    worker.submit_imu([(ImuSample("gyro", 0.0, 0, 0, 1), 2.0)])
    info = worker.stop_clip().result(timeout=5)
    worker.submit_frame(image, 5.0, k)  # after the clip
    worker.close()

    assert info is not None
    assert (info.path, info.frames, info.imu_samples) == (clip_path, 4, 1)
    assert info.seconds == pytest.approx(0.3)
    topics = read_topics(clip_path)
    assert topics[TOPIC_IMAGE] == 4
    assert topics[TOPIC_META] == 1  # repeated so the clip stands alone
    assert topics[TOPIC_CALIBRATION] == 1
    assert not worker.clip_active


def test_close_saves_running_clip(tmp_path: Path) -> None:
    worker = RecordingWorker(Telemetry(), None, jpeg_quality=80, max_queue=100)
    worker.start_clip(tmp_path / "c.mcap").result(timeout=5)
    worker.submit_frame(np.zeros((8, 8, 3), np.uint8), 1.0, Intrinsics.from_fov(8, 8, 60.0))
    info = worker.close()
    assert info is not None
    assert info.frames == 1
    assert read_topics(info.path)[TOPIC_IMAGE] == 1


def test_stop_without_clip_returns_none() -> None:
    worker = RecordingWorker(Telemetry(), None, jpeg_quality=80, max_queue=10)
    assert worker.stop_clip().result(timeout=5) is None
    assert worker.close() is None
