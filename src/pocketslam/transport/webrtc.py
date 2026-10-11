"""WebRTC receive side: answer a phone's offer and deliver frames, metadata and IMU.

This module knows nothing about depth, SLAM or recording. It hands decoded
frames, session metadata and motion-sensor samples to callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
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

from pocketslam.core.types import ImuSample, Intrinsics, LinkStats, RecordRequest, SessionMetadata

logger = logging.getLogger(__name__)

MAX_DIMENSION = 8192
# Decoded frames waiting in aiortc's queue before we skip to the newest one
# (3 frames is about 100 ms at 30 fps).
MAX_FRAME_BACKLOG = 3
MAX_IMU_SAMPLES_PER_MESSAGE = 500
IMU_KINDS = {"g": "gyro", "a": "accel"}

FrameCallback = Callable[[NDArray[np.uint8], float], None]
MetadataCallback = Callable[[SessionMetadata], None]
ImuCallback = Callable[[list[ImuSample]], None]
StatsCallback = Callable[[LinkStats], None]
ControlCallback = Callable[[RecordRequest], None]
_STATS_FIELDS = {f.name: f.type for f in fields(LinkStats)}


class MessageError(ValueError):
    """A DataChannel message is malformed or too large."""


def parse_message(
    raw: str | bytes, max_bytes: int
) -> SessionMetadata | list[ImuSample] | LinkStats | RecordRequest:
    """Parse and validate one DataChannel message from the phone.

    Two message types are accepted. Session metadata::

        {"type": "meta", "width": 1280, "height": 720, "device": "...",
         "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}}

    ``intrinsics`` is optional. A batch of motion-sensor samples::

        {"type": "imu", "samples": [["g", t_ms, x, y, z], ["a", t_ms, x, y, z]]}

    ``"g"`` is gyroscope in rad/s, ``"a"`` is accelerometer (with gravity) in
    m/s^2, and ``t_ms`` is the phone's Unix time in milliseconds. Sender
    diagnostics, once a second::

        {"type": "stats", "camera_fps": 15, "sent_fps": 4, "quality_limitation": "bandwidth", ...}

    Unknown stats fields are ignored; see ``LinkStats`` for the known ones.
    Clip recording control from the phone's Record button::

        {"type": "record", "on": true}

    Args:
        raw: Message payload.
        max_bytes: Reject messages larger than this.

    Returns:
        Session metadata, the IMU samples in the batch, link stats, or a record request.

    Raises:
        MessageError: If the message is too big, not JSON, or has bad fields.
    """
    size = len(raw.encode() if isinstance(raw, str) else raw)
    if size > max_bytes:
        msg = f"message is {size} bytes, limit is {max_bytes}"
        raise MessageError(msg)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = "message is not valid JSON"
        raise MessageError(msg) from exc
    if not isinstance(data, dict):
        msg = "message must be a JSON object"
        raise MessageError(msg)
    if data.get("type") == "meta":
        return _parse_meta(data)
    if data.get("type") == "imu":
        return _parse_imu(data)
    if data.get("type") == "stats":
        return _parse_stats(data)
    if data.get("type") == "record":
        if not isinstance(data.get("on"), bool):
            msg = "record message needs a boolean 'on'"
            raise MessageError(msg)
        return RecordRequest(on=data["on"])
    msg = "message type must be 'meta', 'imu', 'stats' or 'record'"
    raise MessageError(msg)


def _parse_meta(data: dict[str, Any]) -> SessionMetadata:
    width = _positive_int(data, "width")
    height = _positive_int(data, "height")
    device = str(data.get("device", "unknown"))[:200]

    intrinsics = None
    k = data.get("intrinsics")
    if k is not None:
        if not isinstance(k, dict):
            msg = "intrinsics must be an object"
            raise MessageError(msg)
        try:
            fx, fy, cx, cy = (float(k[name]) for name in ("fx", "fy", "cx", "cy"))
        except (KeyError, TypeError, ValueError) as exc:
            msg = "intrinsics need numeric fx, fy, cx, cy"
            raise MessageError(msg) from exc
        if fx <= 0 or fy <= 0:
            msg = "focal lengths must be positive"
            raise MessageError(msg)
        intrinsics = Intrinsics(fx, fy, cx, cy, width, height, source="device")

    orientation = data.get("orientation", 0)
    if orientation not in (0, 90, 180, 270):
        msg = "orientation must be 0, 90, 180 or 270"
        raise MessageError(msg)
    return SessionMetadata(
        width=width,
        height=height,
        device_label=device,
        intrinsics=intrinsics,
        orientation_deg=orientation,
        model=str(data["model"])[:80] if isinstance(data.get("model"), str) else None,
    )


def _parse_stats(data: dict[str, Any]) -> LinkStats:
    values: dict[str, Any] = {}
    for name, kind in _STATS_FIELDS.items():
        value = data.get(name)
        if value is None:
            continue
        if "str" in str(kind):
            if not isinstance(value, str):
                msg = f"stats field {name} must be a string"
                raise MessageError(msg)
            values[name] = value[:80]
        else:
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"stats field {name} must be a number"
                raise MessageError(msg)
            if not math.isfinite(value):
                msg = f"stats field {name} must be finite"
                raise MessageError(msg)
            values[name] = int(value) if "int" in str(kind) else float(value)
    return LinkStats(**values)


def _parse_imu(data: dict[str, Any]) -> list[ImuSample]:
    rows = data.get("samples")
    if not isinstance(rows, list) or len(rows) > MAX_IMU_SAMPLES_PER_MESSAGE:
        msg = f"samples must be a list of at most {MAX_IMU_SAMPLES_PER_MESSAGE} rows"
        raise MessageError(msg)
    samples = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 5 or row[0] not in IMU_KINDS:  # noqa: PLR2004 - row layout
            msg = "each sample must be [kind, t_ms, x, y, z] with kind 'g' or 'a'"
            raise MessageError(msg)
        values = row[1:]
        if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
            msg = "sample time and values must be numbers"
            raise MessageError(msg)
        t_ms, x, y, z = (float(v) for v in values)
        if not all(math.isfinite(v) for v in (t_ms, x, y, z)):
            msg = "sample time and values must be finite"
            raise MessageError(msg)
        samples.append(ImuSample(IMU_KINDS[row[0]], t_ms / 1000.0, x, y, z))
    return samples


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= MAX_DIMENSION:
        msg = f"{key} must be an integer in 1..{MAX_DIMENSION}"
        raise MessageError(msg)
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
    channel: RTCDataChannel | None = None
    frames_skipped: int = 0

    def send_json(self, data: dict[str, Any]) -> bool:
        """Send a JSON message to the phone; False if the channel is not open."""
        if self.channel is None or self.channel.readyState != "open":
            return False
        self.channel.send(json.dumps(data))
        return True

    async def close(self) -> None:
        """Close the peer connection and stop its tracks."""
        await self.pc.close()


async def accept_offer(
    sdp: str,
    sdp_type: str,
    *,
    on_frame: FrameCallback,
    on_metadata: MetadataCallback,
    on_imu: ImuCallback,
    on_closed: Callable[[], None],
    on_stats: StatsCallback | None = None,
    on_control: ControlCallback | None = None,
    max_message_bytes: int,
) -> PeerSession:
    """Answer a phone's SDP offer and start receiving its camera track.

    Args:
        sdp: Offer SDP from the phone.
        sdp_type: Offer type, must be ``"offer"``.
        on_frame: Called with each ``(H, W, 3)`` uint8 RGB image and its sender
            timestamp in seconds.
        on_metadata: Called each time valid metadata arrives.
        on_imu: Called with each batch of valid IMU samples.
        on_closed: Called once when the connection fails or closes.
        on_stats: Called with each link-stats report from the phone.
        on_control: Called when the phone presses Record or Stop recording.
        max_message_bytes: DataChannel message size limit.

    Returns:
        The session, holding the answer to return to the phone.
    """
    # No STUN: phone and host share a LAN, and aiortc otherwise queries Google's
    # public STUN server on every connection.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    closed = False
    peer = PeerSession(pc=pc, answer=RTCSessionDescription(sdp="", type="answer"))

    def _closed_once() -> None:
        nonlocal closed
        if not closed:
            closed = True
            on_closed()

    @pc.on("datachannel")
    def _on_datachannel(channel: RTCDataChannel) -> None:
        peer.channel = channel

        @channel.on("message")
        def _on_message(message: str | bytes) -> None:
            try:
                parsed = parse_message(message, max_message_bytes)
            except MessageError as exc:
                logger.warning("rejected datachannel message", extra={"reason": str(exc)})
                return
            if isinstance(parsed, SessionMetadata):
                on_metadata(parsed)
            elif isinstance(parsed, LinkStats):
                if on_stats is not None:
                    on_stats(parsed)
            elif isinstance(parsed, RecordRequest):
                if on_control is not None:
                    on_control(parsed)
            else:
                on_imu(parsed)

    @pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        if track.kind != "video":
            return
        task = asyncio.create_task(_receive_video(track, on_frame, peer))
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
    peer.answer = pc.localDescription
    return peer


def skip_to_newest(track: MediaStreamTrack, frame: Any) -> tuple[Any, int]:
    """Drop stale frames when a backlog has built up, keeping only the newest.

    aiortc queues every decoded frame without limit. If this process falls
    behind even briefly, the backlog never drains at a steady frame rate and
    the video stays seconds late for the rest of the session. A live pipeline
    needs the newest frame, so anything older is skipped.

    Args:
        track: The receiving track.
        frame: The frame just taken from it.

    Returns:
        ``(frame to use, frames skipped)``.
    """
    queue = getattr(track, "_queue", None)
    if queue is None or queue.qsize() < MAX_FRAME_BACKLOG:
        return frame, 0
    skipped = 0
    while not queue.empty():
        newer = queue.get_nowait()
        if newer is None:  # end-of-stream marker: leave it for recv() to handle
            queue.put_nowait(None)
            break
        frame = newer
        skipped += 1
    return frame, skipped


async def _receive_video(
    track: MediaStreamTrack, on_frame: FrameCallback, peer: PeerSession | None = None
) -> None:
    """Decode frames from ``track`` until it ends."""
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            return
        frame, skipped = skip_to_newest(track, frame)
        if skipped and peer is not None:
            if peer.frames_skipped == 0:
                logger.warning("video fell behind; skipping to the newest frame")
            peer.frames_skipped += skipped
        if not isinstance(frame, VideoFrame):
            continue
        # frame.time is pts * time_base: the sender's RTP clock, not arrival time.
        timestamp = float(frame.time) if frame.time is not None else 0.0
        # rgb24 always decodes to (H, W, 3) uint8; av types the result as a union.
        image = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
        on_frame(image, timestamp)
