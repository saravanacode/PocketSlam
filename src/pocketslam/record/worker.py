"""Background thread that encodes frames and logs them to Foxglove sinks.

JPEG encoding takes several milliseconds per frame, so it runs off the
asyncio event loop. The queue is bounded: if encoding falls behind, new
items are dropped and counted rather than blocking the receiver.

Clip recording (the Record button) is controlled through the same queue, so
a clip contains exactly the items submitted between start and stop.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import foxglove
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pocketslam.core.types import ClipInfo, ImuSample, Intrinsics, SessionMetadata
from pocketslam.record.telemetry import Telemetry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FrameItem:
    image: NDArray[np.uint8]
    t_s: float
    intrinsics: Intrinsics


@dataclass(frozen=True, slots=True)
class _ImuItem:
    samples: list[tuple[ImuSample, float]]


@dataclass(frozen=True, slots=True)
class _MetaItem:
    meta: SessionMetadata
    t_s: float


@dataclass(frozen=True, slots=True)
class _ClipItem:
    path: Path | None  # None stops the current clip
    done: Future[ClipInfo | None]


_Item = _FrameItem | _ImuItem | _MetaItem | _ClipItem | None
CONTROL_TIMEOUT_S = 5.0


class LiveSink(Protocol):
    """Receives recorded data for live display. Called from the worker thread."""

    def publish_frame(self, jpeg: bytes, t_s: float) -> None:
        """Handle one encoded camera frame at host time ``t_s``."""
        ...

    def publish_imu(self, samples: list[tuple[ImuSample, float]]) -> None:
        """Handle IMU samples paired with host times."""
        ...


def encode_jpeg(image: NDArray[np.uint8], quality: int) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB image as JPEG."""
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class RecordingWorker:
    """Logs one phone session to the live viewer and, optionally, an MCAP file."""

    def __init__(
        self,
        telemetry: Telemetry,
        path: Path | None,
        jpeg_quality: int,
        max_queue: int,
        live: LiveSink | None = None,
    ) -> None:
        self.path = path
        self.frames_written = 0
        self.imu_written = 0
        self.dropped = 0
        self._telemetry = telemetry
        self._live = live
        self._quality = jpeg_quality
        self._queue: queue.Queue[_Item] = queue.Queue(maxsize=max_queue)
        self._writer: foxglove.MCAPWriter | None = (
            telemetry.open_recording(path) if path is not None else None
        )
        self._last_intrinsics: Intrinsics | None = None
        self._last_meta: _MetaItem | None = None
        self._clip_writer: foxglove.MCAPWriter | None = None
        self.clip_path: Path | None = None
        self.clip_frames = 0
        self._clip_imu = 0
        self._clip_first_t: float | None = None
        self._clip_last_t: float | None = None
        self._thread = threading.Thread(target=self._run, name="recording-worker", daemon=True)
        self._thread.start()

    def submit_frame(self, image: NDArray[np.uint8], t_s: float, intrinsics: Intrinsics) -> None:
        """Queue a frame captured at host time ``t_s``."""
        self._put(_FrameItem(image, t_s, intrinsics))

    def submit_imu(self, samples: list[tuple[ImuSample, float]]) -> None:
        """Queue IMU samples, each paired with its host time in seconds."""
        if samples:
            self._put(_ImuItem(samples))

    def submit_meta(self, meta: SessionMetadata, t_s: float) -> None:
        """Queue the phone's session metadata."""
        self._put(_MetaItem(meta, t_s))

    def start_clip(self, path: Path) -> Future[ClipInfo | None]:
        """Start writing a clip to ``path``; stops a running clip first.

        Returns:
            Resolves when the file is open (result None), or with the previous
            clip if one was running.
        """
        done: Future[ClipInfo | None] = Future()
        self._queue.put(_ClipItem(path, done), timeout=CONTROL_TIMEOUT_S)
        return done

    def stop_clip(self) -> Future[ClipInfo | None]:
        """Finish the current clip once every item queued before this call is written.

        Returns:
            Resolves with the finished clip, or None if no clip was running.
        """
        done: Future[ClipInfo | None] = Future()
        self._queue.put(_ClipItem(None, done), timeout=CONTROL_TIMEOUT_S)
        return done

    @property
    def clip_active(self) -> bool:
        """True while a clip is being written."""
        return self.clip_path is not None

    def _put(self, item: _Item) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            if isinstance(item, _ClipItem):
                self._handle_clip(item)
                continue
            try:
                self._handle(item)
            except Exception:
                logger.exception("failed to record item")

    def _finish_clip(self) -> ClipInfo | None:
        if self._clip_writer is None or self.clip_path is None:
            return None
        self._clip_writer.close()
        first, last = self._clip_first_t, self._clip_last_t
        info = ClipInfo(
            path=self.clip_path,
            frames=self.clip_frames,
            imu_samples=self._clip_imu,
            seconds=(last - first) if first is not None and last is not None else 0.0,
        )
        self._clip_writer = None
        self.clip_path = None
        logger.info(
            "clip saved",
            extra={
                "path": str(info.path),
                "frames": info.frames,
                "seconds": round(info.seconds, 2),
            },
        )
        return info

    def _handle_clip(self, item: _ClipItem) -> None:
        try:
            previous = self._finish_clip()
            if item.path is not None:
                self._clip_writer = self._telemetry.open_recording(item.path)
                self.clip_path = item.path
                self.clip_frames = self._clip_imu = 0
                self._clip_first_t = self._clip_last_t = None
                # A clip must stand alone: repeat metadata, and log calibration
                # again with the next frame.
                self._last_intrinsics = None
                if self._last_meta is not None:
                    self._telemetry.log_meta(self._last_meta.meta, self._last_meta.t_s)
                logger.info("clip started", extra={"path": str(item.path)})
            item.done.set_result(previous)
        except Exception as exc:
            logger.exception("clip control failed")
            item.done.set_exception(exc)

    def _handle(self, item: _FrameItem | _ImuItem | _MetaItem) -> None:
        if isinstance(item, _FrameItem):
            if item.intrinsics != self._last_intrinsics:
                self._telemetry.log_calibration(item.intrinsics, item.t_s)
                self._last_intrinsics = item.intrinsics
            jpeg = encode_jpeg(item.image, self._quality)
            self._telemetry.log_image(jpeg, item.t_s)
            self.frames_written += 1
            if self.clip_path is not None:
                self.clip_frames += 1
                self._clip_first_t = (
                    self._clip_first_t if self._clip_first_t is not None else item.t_s
                )
                self._clip_last_t = item.t_s
            if self._live is not None:
                self._live.publish_frame(jpeg, item.t_s)
        elif isinstance(item, _ImuItem):
            for sample, t_s in item.samples:
                self._telemetry.log_imu(sample, t_s)
            self.imu_written += len(item.samples)
            if self.clip_path is not None:
                self._clip_imu += len(item.samples)
            if self._live is not None:
                self._live.publish_imu(item.samples)
        else:
            self._last_meta = item
            self._telemetry.log_meta(item.meta, item.t_s)

    def close(self) -> ClipInfo | None:
        """Drain the queue, stop the thread and finish the MCAP files.

        Returns:
            The clip that was still recording, if any (it is saved, not lost).
        """
        self._queue.put(None)
        self._thread.join()
        clip = self._finish_clip()
        if self._writer is not None:
            self._writer.close()
        logger.info(
            "recording closed",
            extra={
                "path": str(self.path) if self.path else None,
                "frames": self.frames_written,
                "imu_samples": self.imu_written,
                "dropped": self.dropped,
            },
        )
        return clip
