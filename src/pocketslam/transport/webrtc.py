"""WebRTC receive side: answer a phone's offer and deliver frames and metadata.

This module knows nothing about depth or SLAM. It hands decoded frames and
parsed session metadata to callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
from numpy.typing import NDArray

from pocketslam.core.types import Intrinsics, SessionMetadata

logger = logging.getLogger(__name__)

MAX_DIMENSION = 8192

FrameCallback = Callable[[NDArray[np.uint8], float], None]
MetadataCallback = Callable[[SessionMetadata], None]


class MetadataError(ValueError):
    """A DataChannel message is not valid session metadata."""


def parse_metadata(raw: str | bytes, max_bytes: int) -> SessionMetadata:
    """Parse and validate a session metadata message from the phone.

    Expected JSON::

        {"type": "meta", "width": 1280, "height": 720, "device": "...",
         "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}}

    ``intrinsics`` is optional.

    Args:
        raw: Message payload.
        max_bytes: Reject messages larger than this.

    Returns:
        The parsed metadata.

    Raises:
        MetadataError: If the message is too big, not JSON, or has bad fields.
    """
    size = len(raw.encode() if isinstance(raw, str) else raw)
    if size > max_bytes:
        msg = f"metadata message is {size} bytes, limit is {max_bytes}"
        raise MetadataError(msg)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = "metadata message is not valid JSON"
        raise MetadataError(msg) from exc
    if not isinstance(data, dict) or data.get("type") != "meta":
        msg = "metadata message must be an object with type 'meta'"
        raise MetadataError(msg)

    width = _positive_int(data, "width")
    height = _positive_int(data, "height")
    device = str(data.get("device", "unknown"))[:200]

    intrinsics = None
    k = data.get("intrinsics")
    if k is not None:
        if not isinstance(k, dict):
            msg = "intrinsics must be an object"
            raise MetadataError(msg)
        try:
            fx, fy, cx, cy = (float(k[name]) for name in ("fx", "fy", "cx", "cy"))
        except (KeyError, TypeError, ValueError) as exc:
            msg = "intrinsics need numeric fx, fy, cx, cy"
            raise MetadataError(msg) from exc
        if fx <= 0 or fy <= 0:
            msg = "focal lengths must be positive"
            raise MetadataError(msg)
        intrinsics = Intrinsics(fx, fy, cx, cy, width, height, source="device")

    return SessionMetadata(width=width, height=height, device_label=device, intrinsics=intrinsics)


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= MAX_DIMENSION:
        msg = f"{key} must be an integer in 1..{MAX_DIMENSION}"
        raise MetadataError(msg)
    return value


@dataclass
class PeerSession:
    """A connected phone.

    Attributes:
        pc: The peer connection.
        answer: The SDP answer to send back to the phone.
    """

    pc: RTCPeerConnection
    answer: RTCSessionDescription

    async def close(self) -> None:
        """Close the peer connection and stop its tracks."""
        await self.pc.close()


async def accept_offer(
    sdp: str,
    sdp_type: str,
    *,
    on_frame: FrameCallback,
    on_metadata: MetadataCallback,
    on_closed: Callable[[], None],
    max_message_bytes: int,
) -> PeerSession:
    """Answer a phone's SDP offer and start receiving its camera track.

    Args:
        sdp: Offer SDP from the phone.
        sdp_type: Offer type, must be ``"offer"``.
        on_frame: Called with each ``(H, W, 3)`` uint8 RGB image and its sender
            timestamp in seconds.
        on_metadata: Called each time valid metadata arrives.
        on_closed: Called once when the connection fails or closes.
        max_message_bytes: DataChannel message size limit.

    Returns:
        The session, holding the answer to return to the phone.
    """
    # No STUN: phone and host share a LAN, and aiortc otherwise queries Google's
    # public STUN server on every connection.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    closed = False

    def _closed_once() -> None:
        nonlocal closed
        if not closed:
            closed = True
            on_closed()

    @pc.on("datachannel")
    def _on_datachannel(channel: RTCDataChannel) -> None:
        @channel.on("message")
        def _on_message(message: str | bytes) -> None:
            try:
                on_metadata(parse_metadata(message, max_message_bytes))
            except MetadataError as exc:
                logger.warning("rejected datachannel message", extra={"reason": str(exc)})

    @pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        if track.kind != "video":
            return
        task = asyncio.create_task(_receive_video(track, on_frame))
        task.add_done_callback(lambda _: _closed_once())

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        logger.info("peer connection state", extra={"state": pc.connectionState})
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            _closed_once()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return PeerSession(pc=pc, answer=pc.localDescription)


async def _receive_video(track: MediaStreamTrack, on_frame: FrameCallback) -> None:
    """Decode frames from ``track`` until it ends."""
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            return
        if not isinstance(frame, VideoFrame):
            continue
        # frame.time is pts * time_base: the sender's RTP clock, not arrival time.
        timestamp = float(frame.time) if frame.time is not None else 0.0
        # rgb24 always decodes to (H, W, 3) uint8; av types the result as a union.
        image = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
        on_frame(image, timestamp)
