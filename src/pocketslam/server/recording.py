"""Clip recording (the Record button) and automatic reconstruction of saved clips.

``RecordingControl`` starts and stops clips on the active phone session, from
the phone's button or the dashboard's. Each saved clip goes to
``ReconJobs``, which runs ``pocketslam-recon`` on it in a separate process, one
clip at a time. A separate process keeps the 3 GB of model weights and the GPU
work out of the live server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pocketslam.config import Settings
from pocketslam.core.types import ClipInfo, RecordRequest
from pocketslam.pipeline import Session

logger = logging.getLogger(__name__)

MIN_CLIP_FRAMES = 10
MAX_JOBS_KEPT = 20
LOG_TAIL_LINES = 15


@dataclass
class ReconJob:
    """One clip waiting for, or going through, reconstruction.

    Attributes:
        name: Scan name; also the output folder under ``reconstructions_dir``.
        clip: The clip's MCAP file.
        frames: Frames in the clip.
        seconds: Clip length.
        state: ``queued``, ``running``, ``done``, ``failed`` or ``skipped``.
        message: Why it failed or was skipped.
        queued_at: Unix time it was queued.
        finished_at: Unix time it ended.
    """

    name: str
    clip: str
    frames: int
    seconds: float
    state: str = "queued"
    message: str = ""
    queued_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class ReconJobs:
    """Runs ``pocketslam-recon`` on saved clips, one at a time, in a subprocess."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: deque[ReconJob] = deque(maxlen=MAX_JOBS_KEPT)
        self._queue: asyncio.Queue[ReconJob] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self.command: list[str] = [sys.executable, "-m", "pocketslam.recon.cli"]

    def start(self) -> None:
        """Start the worker task (call from the running event loop)."""
        self._task = asyncio.create_task(self._run(), name="recon-jobs")

    async def stop(self) -> None:
        """Stop after the current job is interrupted."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def submit(self, clip: ClipInfo) -> ReconJob:
        """Queue a saved clip; very short clips are recorded as skipped."""
        job = ReconJob(
            name=clip.path.stem, clip=str(clip.path), frames=clip.frames, seconds=clip.seconds
        )
        if clip.frames < MIN_CLIP_FRAMES:
            job.state = "skipped"
            job.message = f"clip has {clip.frames} frames; at least {MIN_CLIP_FRAMES} are needed"
            job.finished_at = time.time()
        else:
            self._queue.put_nowait(job)
        self._jobs.append(job)
        logger.info("reconstruction queued", extra={"job": job.name, "state": job.state})
        return job

    def snapshot(self) -> list[dict[str, Any]]:
        """Jobs, newest first, as JSON-ready dicts."""
        return [asdict(j) for j in reversed(self._jobs)]

    @property
    def busy(self) -> bool:
        """True while a job is queued or running."""
        return any(j.state in {"queued", "running"} for j in self._jobs)

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._run_job(job)
            except Exception as exc:  # one bad job must not stop later clips
                logger.exception("reconstruction job crashed", extra={"job": job.name})
                job.state, job.message = "failed", f"internal error: {exc}"
                job.finished_at = job.finished_at or time.time()

    async def _run_job(self, job: ReconJob) -> None:
        """Run ``pocketslam-recon`` on one clip and record the outcome on ``job``."""
        job.state = "running"
        out = self._settings.reconstructions_dir / job.name
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                job.clip,
                "--out",
                str(out),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await proc.communicate()
            if proc.returncode == 0:
                job.state = "done"
            else:
                tail = output.decode(errors="replace").strip().splitlines()[-LOG_TAIL_LINES:]
                job.state = "failed"
                job.message = _explain_failure(tail)
        except OSError as exc:
            job.state, job.message = "failed", f"could not start reconstruction: {exc}"
        job.finished_at = time.time()
        logger.info(
            "reconstruction finished",
            extra={
                "job": job.name,
                "state": job.state,
                "seconds": round(time.perf_counter() - started, 1),
                "detail": job.message,
            },
        )


def _explain_failure(tail: list[str]) -> str:
    text = "\n".join(tail)
    if "uv sync --extra recon" in text:
        return (
            "Depth Anything 3 is not installed. "
            "Run: uv sync --extra recon, then restart the server."
        )
    return tail[-1] if tail else "reconstruction failed without output"


class RecordingControl:
    """Start and stop clips on the active session and keep the phone informed."""

    def __init__(self, settings: Settings, jobs: ReconJobs | None) -> None:
        self._settings = settings
        self._jobs = jobs

    def clip_path(self) -> Path:
        """New clip file name, unique to the second."""
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        return self._settings.recordings_dir / f"clip_{stamp}.mcap"

    def attach(self, session: Session) -> None:
        """Wire a new session's Record button and saved clips to this controller."""
        session.on_clip_saved = self._clip_saved
        session.on_control = self._phone_request

    async def start(self, session: Session) -> Path:
        """Start a clip on ``session`` and return its file path."""
        path = self.clip_path()
        await session.start_clip(path)
        self._notify(session)
        return path

    async def stop(self, session: Session) -> ClipInfo | None:
        """Stop the clip on ``session``; the saved clip is queued for reconstruction."""
        info = await session.stop_clip()
        self._notify(session, saved=info)
        return info

    def _clip_saved(self, clip: ClipInfo) -> None:
        if self._jobs is not None and self._settings.auto_reconstruct:
            self._jobs.submit(clip)

    def _phone_request(self, session: Session, request: RecordRequest) -> None:
        action = self.start(session) if request.on else self.stop(session)
        task = asyncio.get_running_loop().create_task(action)
        task.add_done_callback(_log_task_error)

    def _notify(self, session: Session, saved: ClipInfo | None = None) -> None:
        clip = session.stats()["clip"]
        message: dict[str, object] = {"type": "record_state", "on": clip is not None, "clip": clip}
        if saved is not None:
            message["saved"] = {
                "file": saved.path.name,
                "frames": saved.frames,
                "seconds": round(saved.seconds, 2),
            }
        session.notify_phone(message)


def _log_task_error(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error("record control failed", exc_info=task.exception())
