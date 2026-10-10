"""Per-phone session: wires transport, depth, SLAM and visualisation together."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from pocketslam.config import Settings
from pocketslam.core.types import Frame, Intrinsics, SessionMetadata
from pocketslam.depth.base import ConstantDepthEstimator, DepthEstimator
from pocketslam.slam.base import SlamBackend, StaticSlam
from pocketslam.transport.frame_queue import DropOldestQueue
from pocketslam.transport.webrtc import PeerSession, accept_offer
from pocketslam.viz.base import TrajectoryWriter, Visualizer

logger = logging.getLogger(__name__)

STATS_EVERY_N_FRAMES = 30


class Session:
    """Processes frames from one phone, newest first, off the event loop."""

    def __init__(
        self,
        session_id: str,
        settings: Settings,
        depth: DepthEstimator,
        slam: SlamBackend,
        viz: Visualizer,
    ) -> None:
        self.session_id = session_id
        self.settings = settings
        self.queue: DropOldestQueue[Frame] = DropOldestQueue(settings.queue_size)
        self.metadata: SessionMetadata | None = None
        self.processed = 0
        self._depth = depth
        self._slam = slam
        self._viz = viz
        self._intrinsics: Intrinsics | None = None
        self._next_index = 0
        self._task: asyncio.Task[None] | None = None
        self._peer: PeerSession | None = None
        self._log = logging.LoggerAdapter(logger, {"session_id": session_id})

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

    def on_frame(self, image: NDArray[np.uint8], timestamp_s: float) -> None:
        """Wrap a decoded image in a ``Frame`` and enqueue it."""
        height, width = int(image.shape[0]), int(image.shape[1])
        if self._intrinsics is None:
            self._intrinsics = Intrinsics.from_fov(width, height, self.settings.default_hfov_deg)
            self._log.warning(
                "no intrinsics from phone, using FOV estimate",
                extra={"hfov_deg": self.settings.default_hfov_deg},
            )
        intrinsics = self._intrinsics.scaled(width, height)
        self.queue.put(Frame(image, timestamp_s, self._next_index, intrinsics))
        self._next_index += 1

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
                        "tracking": pose is not None,
                    },
                )
                window_start = time.perf_counter()
                latency_sum = 0.0

    async def close(self) -> None:
        """Stop receiving, drain the loop and flush outputs."""
        if self._peer is not None:
            await self._peer.close()
        self.queue.close()
        if self._task is not None:
            await self._task
        self._viz.close()
        self._log.info("session closed", extra={"frames": self.processed})


def create_session(settings: Settings) -> Session:
    """Build a session with the configured backends.

    The placeholder backends below are swapped for real depth and SLAM
    backends as they are implemented.
    """
    session_id = uuid.uuid4().hex[:12]
    out_dir = settings.output_dir / session_id
    return Session(
        session_id=session_id,
        settings=settings,
        depth=ConstantDepthEstimator(),
        slam=StaticSlam(),
        viz=TrajectoryWriter(out_dir / "trajectory.tum"),
    )


async def start_session(
    settings: Settings,
    sdp: str,
    sdp_type: str,
    on_closed: Callable[[Session], None] | None = None,
) -> tuple[Session, PeerSession]:
    """Create a session, answer the phone's offer, and start processing.

    Args:
        settings: Runtime settings.
        sdp: Offer SDP from the phone.
        sdp_type: Offer type.
        on_closed: Called with the session when the phone disconnects.

    Returns:
        The running session and its peer, which holds the SDP answer.
    """
    session = create_session(settings)

    def _closed() -> None:
        session.queue.close()
        if on_closed is not None:
            on_closed(session)

    peer = await accept_offer(
        sdp,
        sdp_type,
        on_frame=session.on_frame,
        on_metadata=session.on_metadata,
        on_closed=_closed,
        max_message_bytes=settings.max_message_bytes,
    )
    session.attach_peer(peer)
    session.start()
    return session, peer
