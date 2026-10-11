"""Per-phone session: wires transport, recording, depth, SLAM and visualisation together."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pocketslam.config import Settings
from pocketslam.core.clock import ClockMapper
from pocketslam.core.types import (
    ClipInfo,
    Frame,
    ImuSample,
    Intrinsics,
    LinkStats,
    RecordRequest,
    SessionMetadata,
)
from pocketslam.depth.base import ConstantDepthEstimator, DepthEstimator
from pocketslam.log import SessionLogger
from pocketslam.record.telemetry import Telemetry
from pocketslam.record.worker import LiveSink, RecordingWorker
from pocketslam.slam.base import SlamBackend, StaticSlam
from pocketslam.transport.frame_queue import DropOldestQueue
from pocketslam.transport.webrtc import PeerSession, accept_offer
from pocketslam.viz.base import TrajectoryWriter, Visualizer

logger = logging.getLogger(__name__)

STATS_EVERY_N_FRAMES = 30
LINK_LOG_EVERY_N_REPORTS = 5


class Session:
    """Processes frames from one phone, newest first, off the event loop."""

    def __init__(
        self,
        session_id: str,
        settings: Settings,
        *,
        depth: DepthEstimator,
        slam: SlamBackend,
        viz: Visualizer,
        recorder: RecordingWorker | None = None,
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.queue: DropOldestQueue[Frame] = DropOldestQueue(settings.queue_size)
        self.metadata: SessionMetadata | None = None
        self.processed = 0
        self.imu_received = 0
        self.started_at = time.time()
        self.link: LinkStats | None = None
        self.clip_started_at: float | None = None
        # Set by the server: what to do with a saved clip, and how to handle the
        # phone's Record button.
        self.on_clip_saved: Callable[[ClipInfo], None] | None = None
        self.on_control: Callable[[Session, RecordRequest], None] | None = None
        # Set by the server: receives every decoded frame (live camera calibration).
        self.frame_tap: Callable[[NDArray[np.uint8], float], None] | None = None
        self._link_reports = 0
        self._arrivals: deque[float] = deque(maxlen=120)
        self.recorder = recorder
        self._video_clock = ClockMapper()
        self._imu_clock = ClockMapper()
        self._depth = depth
        self._slam = slam
        self._viz = viz
        self._intrinsics: Intrinsics | None = None
        self._next_index = 0
        self._task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._peer: PeerSession | None = None
        self._log = SessionLogger(logger, {"session_id": session_id})

    def on_metadata(self, metadata: SessionMetadata) -> None:
        """Store the phone's session metadata."""
        self.metadata = metadata
        self._intrinsics = metadata.intrinsics
        self._log.info(
            "session metadata",
            extra={
                "device_label": metadata.device_label,
                "size": f"{metadata.width}x{metadata.height}",
                "intrinsics_source": metadata.intrinsics.source if metadata.intrinsics else None,
            },
        )
        if self.recorder is not None:
            self.recorder.submit_meta(metadata, time.time())

    def on_imu(self, samples: list[ImuSample]) -> None:
        """Map IMU samples onto the host clock and send them to the recorder."""
        now = time.time()
        for sample in samples:
            self._imu_clock.observe(sample.timestamp_s, now)
        self.imu_received += len(samples)
        if self.recorder is not None:
            self.recorder.submit_imu([(s, self._imu_clock.to_host(s.timestamp_s)) for s in samples])

    def on_stats(self, stats: LinkStats) -> None:
        """Store the phone's sender diagnostics and log them periodically."""
        self.link = stats
        self._link_reports += 1
        if self._link_reports % LINK_LOG_EVERY_N_REPORTS == 1:
            self._log.info(
                "link stats",
                extra={"recv_fps": self.recv_fps()}
                | {k: v for k, v in asdict(stats).items() if v is not None},
            )

    async def start_clip(self, path: Path) -> None:
        """Start recording a clip (the Record button). A running clip is saved first."""
        if self.recorder is None:
            msg = "recording needs telemetry; the server was started without it"
            raise RuntimeError(msg)
        previous = await asyncio.wrap_future(self.recorder.start_clip(path))
        self.clip_started_at = time.time()
        if previous is not None and self.on_clip_saved is not None:
            self.on_clip_saved(previous)

    async def stop_clip(self) -> ClipInfo | None:
        """Finish the clip and hand it to ``on_clip_saved``; None if none was running."""
        if self.recorder is None:
            return None
        info = await asyncio.wrap_future(self.recorder.stop_clip())
        self.clip_started_at = None
        if info is not None and self.on_clip_saved is not None:
            self.on_clip_saved(info)
        return info

    def handle_control(self, request: RecordRequest) -> None:
        """Forward the phone's Record button to the server's handler."""
        if self.on_control is not None:
            self.on_control(self, request)

    def notify_phone(self, data: dict[str, object]) -> bool:
        """Send a JSON message to the phone over the data channel."""
        return self._peer.send_json(data) if self._peer is not None else False

    def recv_fps(self) -> float:
        """Frames per second arriving from the phone over the last second."""
        now = time.monotonic()
        return float(sum(1 for t in self._arrivals if now - t <= 1.0))

    def on_frame(self, image: NDArray[np.uint8], sender_time_s: float) -> None:
        """Record a decoded image and enqueue it for processing.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.
            sender_time_s: Capture time on the sender's RTP clock. It is mapped
                onto the host clock so video and IMU share one time base.
        """
        self._arrivals.append(time.monotonic())
        self._video_clock.observe(sender_time_s, time.time())
        timestamp_s = self._video_clock.to_host(sender_time_s)
        if self.frame_tap is not None:
            self.frame_tap(image, timestamp_s)
        height, width = int(image.shape[0]), int(image.shape[1])
        if self._intrinsics is None:
            self._intrinsics = Intrinsics.from_fov(width, height, self.settings.default_hfov_deg)
            self._log.warning(
                "no intrinsics from phone, using FOV estimate",
                extra={"hfov_deg": self.settings.default_hfov_deg},
            )
        intrinsics = self._intrinsics.scaled(width, height)
        if self.recorder is not None:
            self.recorder.submit_frame(image, timestamp_s, intrinsics)
        self.queue.put(Frame(image, timestamp_s, self._next_index, intrinsics))
        self._next_index += 1

    def stats(self) -> dict[str, object]:
        """Return a snapshot of session counters for the dashboard."""
        meta = self.metadata
        rec = self.recorder
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "device": meta.device_label if meta else None,
            "width": meta.width if meta else None,
            "height": meta.height if meta else None,
            "intrinsics_source": self._intrinsics.source if self._intrinsics else None,
            "frames_received": self._next_index,
            "frames_processed": self.processed,
            "imu_received": self.imu_received,
            "recording": str(rec.path) if rec and rec.path else None,
            "frames_recorded": rec.frames_written if rec else 0,
            "record_dropped": rec.dropped if rec else 0,
            "recv_fps": self.recv_fps(),
            "frames_skipped": self._peer.frames_skipped if self._peer else 0,
            "link": asdict(self.link) if self.link else None,
            "clip": (
                {
                    "path": str(rec.clip_path),
                    "started_at": self.clip_started_at,
                    "frames": rec.clip_frames,
                }
                if rec and rec.clip_active
                else None
            ),
        }

    def attach_peer(self, peer: PeerSession) -> None:
        """Give the session ownership of its peer connection."""
        self._peer = peer

    def start(self) -> None:
        """Start the processing loop."""
        self._task = asyncio.create_task(self._run(), name=f"session-{self.session_id}")

    async def _run(self) -> None:
        window_start = time.perf_counter()
        latency_sum = 0.0
        while (frame := await self.queue.get()) is not None:
            t0 = time.perf_counter()
            depth = await asyncio.to_thread(self._depth.estimate, frame)
            pose = await asyncio.to_thread(self._slam.track, frame, depth)
            self._viz.show(frame, depth, pose)
            self.processed += 1
            latency_sum += time.perf_counter() - t0
            if self.processed % STATS_EVERY_N_FRAMES == 0:
                elapsed = time.perf_counter() - window_start
                self._log.info(
                    "pipeline stats",
                    extra={
                        "fps": round(STATS_EVERY_N_FRAMES / elapsed, 2),
                        "latency_ms": round(1000 * latency_sum / STATS_EVERY_N_FRAMES, 1),
                        "dropped": self.queue.dropped,
                        "imu_samples": self.imu_received,
                        "record_dropped": self.recorder.dropped if self.recorder else None,
                        "tracking": pose is not None,
                    },
                )
                window_start = time.perf_counter()
                latency_sum = 0.0

    async def close(self) -> None:
        """Stop receiving, drain the loop and flush outputs.

        Safe to call more than once or concurrently; every caller waits for
        the same shutdown, so the MCAP file is finished exactly once.
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._peer is not None:
            await self._peer.close()
        self.queue.close()
        if self._task is not None:
            await self._task
        self._viz.close()
        if self.recorder is not None:
            clip = await asyncio.to_thread(self.recorder.close)
            if clip is not None and self.on_clip_saved is not None:
                self.on_clip_saved(clip)
        self._log.info("session closed", extra={"frames": self.processed})


def create_session(
    settings: Settings,
    telemetry: Telemetry | None = None,
    live: LiveSink | None = None,
) -> Session:
    """Build a session with the configured backends.

    The placeholder depth and SLAM backends are swapped for real ones as they
    are implemented.

    Args:
        settings: Runtime settings.
        telemetry: Foxglove channels for MCAP recording and the Foxglove live
            view. None disables recording and live output.
        live: Receives encoded frames and IMU for the laptop dashboard.
    """
    session_id = uuid.uuid4().hex[:12]
    out_dir = settings.output_dir / session_id
    recorder = None
    if telemetry is not None:
        path = None
        if settings.record:
            stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            path = settings.recordings_dir / f"{stamp}_{session_id}.mcap"
        recorder = RecordingWorker(
            telemetry, path, settings.jpeg_quality, settings.record_queue_size, live
        )
    return Session(
        session_id=session_id,
        settings=settings,
        depth=ConstantDepthEstimator(),
        slam=StaticSlam(),
        viz=TrajectoryWriter(out_dir / "trajectory.tum"),
        recorder=recorder,
    )


async def start_session(
    settings: Settings,
    sdp: str,
    sdp_type: str,
    *,
    on_closed: Callable[[Session], None] | None = None,
    telemetry: Telemetry | None = None,
    live: LiveSink | None = None,
) -> tuple[Session, PeerSession]:
    """Create a session, answer the phone's offer, and start processing.

    Args:
        settings: Runtime settings.
        sdp: Offer SDP from the phone.
        sdp_type: Offer type.
        on_closed: Called with the session when the phone disconnects.
        telemetry: Foxglove channels for recording and live view, or None.
        live: Receives encoded frames and IMU for the laptop dashboard.

    Returns:
        The running session and its peer, which holds the SDP answer.
    """
    session = create_session(settings, telemetry, live)

    def _closed() -> None:
        session.queue.close()
        if on_closed is not None:
            on_closed(session)

    peer = await accept_offer(
        sdp,
        sdp_type,
        on_frame=session.on_frame,
        on_metadata=session.on_metadata,
        on_imu=session.on_imu,
        on_closed=_closed,
        on_stats=session.on_stats,
        on_control=session.handle_control,
        max_message_bytes=settings.max_message_bytes,
    )
    session.attach_peer(peer)
    session.start()
    return session, peer
