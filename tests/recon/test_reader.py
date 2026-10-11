from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pocketslam.recon.reader import read_recording
from pocketslam.record.telemetry import Telemetry


def test_reads_frames_imu_and_meta(write_recording: Callable[..., Path]) -> None:
    rec = read_recording(write_recording(frames=12, imu=40))
    assert len(rec) == 12
    assert not rec.truncated
    assert rec.frame_times[0] == pytest.approx(100.0)
    assert rec.duration_s == pytest.approx(11 / 30)
    assert rec.decode(3).shape == (24, 32, 3)
    assert len(rec.gyro.t) == len(rec.accel.t) == 40
    np.testing.assert_allclose(rec.gyro.xyz[0], [0.1, 0.2, 0.3])
    assert rec.meta is not None
    assert rec.meta["width"] == 32


def test_clip_selects_time_window(write_recording: Callable[..., Path]) -> None:
    rec = read_recording(write_recording(frames=30)).clip(start_s=0.2, duration_s=0.3)
    assert rec.frame_times[0] >= 100.2 - 1e-9
    assert rec.frame_times[-1] <= 100.5 + 1e-9
    with pytest.raises(ValueError, match="no frames"):
        rec.clip(start_s=50.0)


def test_reads_cut_off_file(write_recording: Callable[..., Path], tmp_path: Path) -> None:
    full = write_recording(frames=20).read_bytes()
    cut = tmp_path / "cut.mcap"
    cut.write_bytes(full[: len(full) // 2])
    rec = read_recording(cut)
    assert rec.truncated
    assert 0 < len(rec) <= 20


def test_rejects_file_without_frames(tmp_path: Path) -> None:
    path = tmp_path / "empty.mcap"
    Telemetry().open_recording(path).close()
    with pytest.raises(ValueError, match="no camera frames"):
        read_recording(path)
