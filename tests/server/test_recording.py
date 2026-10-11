import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

from pocketslam.config import Settings
from pocketslam.core.types import ClipInfo, RecordRequest
from pocketslam.pipeline import create_session
from pocketslam.record.telemetry import Telemetry
from pocketslam.server.recording import MIN_CLIP_FRAMES, ReconJobs, RecordingControl

FAKE_RECON_OK = "import sys, pathlib; pathlib.Path(sys.argv[3]).mkdir(parents=True)"
FAKE_RECON_FAIL = (
    "import sys; print('Depth Anything 3 is not installed. Run: uv sync --extra recon'); "
    "sys.exit(1)"
)


def clip(tmp_path: Path, frames: int = 50) -> ClipInfo:
    return ClipInfo(path=tmp_path / "clip_x.mcap", frames=frames, imu_samples=100, seconds=5.0)


async def wait_done(jobs: ReconJobs) -> None:
    for _ in range(200):
        if not jobs.busy:
            return
        await asyncio.sleep(0.02)


async def test_runs_reconstruction_for_clip(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"reconstructions_dir": tmp_path / "recon"})
    jobs = ReconJobs(s)
    jobs.command = [sys.executable, "-c", FAKE_RECON_OK]
    jobs.start()
    jobs.submit(clip(tmp_path))
    await wait_done(jobs)
    await jobs.stop()
    [job] = jobs.snapshot()
    assert job["state"] == "done", job
    assert (tmp_path / "recon" / "clip_x").is_dir()


async def test_failed_job_explains_fix(settings: Settings, tmp_path: Path) -> None:
    jobs = ReconJobs(settings)
    jobs.command = [sys.executable, "-c", FAKE_RECON_FAIL]
    jobs.start()
    jobs.submit(clip(tmp_path))
    await wait_done(jobs)
    await jobs.stop()
    [job] = jobs.snapshot()
    assert job["state"] == "failed"
    assert "uv sync --extra recon" in job["message"]


def test_short_clip_is_skipped(settings: Settings, tmp_path: Path) -> None:
    job = ReconJobs(settings).submit(clip(tmp_path, frames=MIN_CLIP_FRAMES - 1))
    assert job.state == "skipped"


async def test_control_saves_clip_and_queues_it(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"recordings_dir": tmp_path / "rec"})
    queued: list[ClipInfo] = []

    class Jobs:
        def submit(self, c: ClipInfo) -> None:
            queued.append(c)

    control = RecordingControl(s, Jobs())  # type: ignore[arg-type]
    session = create_session(s, Telemetry())
    control.attach(session)
    path = await control.start(session)
    assert session.stats()["clip"] is not None
    for i in range(3):
        session.on_frame(np.zeros((16, 16, 3), np.uint8), i / 30)
    info = await control.stop(session)
    await session.close()
    assert info is not None
    assert info.path == path
    assert path.name.startswith("clip_")
    assert info.frames == 3
    assert queued == [info]
    assert session.stats()["clip"] is None


async def test_phone_button_starts_and_stops(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"recordings_dir": tmp_path / "rec"})
    control = RecordingControl(s, None)
    session = create_session(s, Telemetry())
    control.attach(session)
    session.handle_control(RecordRequest(on=True))
    await asyncio.sleep(0.1)
    assert session.stats()["clip"] is not None
    session.handle_control(RecordRequest(on=False))
    await asyncio.sleep(0.1)
    assert session.stats()["clip"] is None
    await session.close()
    assert len(list((tmp_path / "rec").glob("clip_*.mcap"))) == 1


async def test_start_without_telemetry_raises(settings: Settings) -> None:
    session = create_session(settings)
    with pytest.raises(RuntimeError, match="telemetry"):
        await session.start_clip(Path("x.mcap"))
    assert await session.stop_clip() is None
    await session.close()


async def test_jobs_keep_running_after_the_first(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"reconstructions_dir": tmp_path / "recon"})
    jobs = ReconJobs(s)
    jobs.command = [sys.executable, "-c", FAKE_RECON_OK]
    jobs.start()
    for name in ("clip_a", "clip_b"):
        jobs.submit(ClipInfo(path=tmp_path / f"{name}.mcap", frames=50, imu_samples=0, seconds=5.0))
    await wait_done(jobs)
    await jobs.stop()
    assert [j["state"] for j in jobs.snapshot()] == ["done", "done"]
