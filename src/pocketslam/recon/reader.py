"""Read a PocketSlam MCAP recording: camera frames, IMU and phone metadata.

Works on finished files and on files still being written (or cut short by a
crash): records are streamed and an incomplete tail is ignored.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcap.exceptions import McapError
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_protobuf.decoder import DecoderFactory
from numpy.typing import NDArray
from PIL import Image

from pocketslam.core.topics import TOPIC_ACCEL, TOPIC_GYRO, TOPIC_IMAGE, TOPIC_META
from pocketslam.slam.vi_align import ImuSeries

logger = logging.getLogger(__name__)


@dataclass
class Recording:
    """Contents of one recording. Frames stay JPEG-encoded until decoded.

    Attributes:
        path: Source file.
        frame_times: ``(F,)`` frame times in seconds (host clock).
        frame_jpegs: Encoded frames, aligned with ``frame_times``.
        gyro: Gyroscope samples (rad/s).
        accel: Accelerometer samples (m/s^2, with gravity).
        meta: Last ``/phone/meta`` message, if any.
        truncated: True if the file ended mid-record (unfinished recording).
    """

    path: Path
    frame_times: NDArray[np.float64]
    frame_jpegs: list[bytes]
    gyro: ImuSeries
    accel: ImuSeries
    meta: dict[str, Any] | None = None
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.frame_jpegs)

    def decode(self, index: int) -> NDArray[np.uint8]:
        """Decode frame ``index`` to ``(H, W, 3)`` uint8 RGB."""
        with Image.open(io.BytesIO(self.frame_jpegs[index])) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def clip(self, start_s: float = 0.0, duration_s: float | None = None) -> Recording:
        """The part of the recording from ``start_s`` after the first frame, ``duration_s`` long."""
        t0 = float(self.frame_times[0]) + start_s
        t1 = t0 + duration_s if duration_s is not None else np.inf
        keep = (self.frame_times >= t0) & (self.frame_times <= t1)
        if not keep.any():
            msg = f"no frames between {start_s} s and {start_s + (duration_s or 0)} s"
            raise ValueError(msg)

        def cut(s: ImuSeries) -> ImuSeries:
            m = (s.t >= t0 - 0.5) & (s.t <= t1 + 0.5)  # margin for the time-offset search
            return ImuSeries(s.t[m], s.xyz[m])

        return Recording(
            path=self.path,
            frame_times=self.frame_times[keep],
            frame_jpegs=[j for j, k in zip(self.frame_jpegs, keep, strict=True) if k],
            gyro=cut(self.gyro),
            accel=cut(self.accel),
            meta=self.meta,
            truncated=self.truncated,
        )

    @property
    def duration_s(self) -> float:
        """Time between the first and last frame."""
        return float(self.frame_times[-1] - self.frame_times[0]) if len(self) > 1 else 0.0


def _series(rows: list[tuple[float, float, float, float]]) -> ImuSeries:
    if not rows:
        return ImuSeries(np.zeros(0), np.zeros((0, 3)))
    arr = np.array(sorted(rows), dtype=np.float64)
    return ImuSeries(arr[:, 0], arr[:, 1:])


def read_recording(path: Path) -> Recording:
    """Load frames, IMU and metadata from ``path``.

    Raises:
        ValueError: If the file holds no camera frames.
    """
    decoders = DecoderFactory()
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    frames: list[tuple[float, bytes]] = []
    gyro: list[tuple[float, float, float, float]] = []
    accel: list[tuple[float, float, float, float]] = []
    meta: dict[str, Any] | None = None
    truncated = False
    image_decoder = None

    with path.open("rb") as f:
        try:
            for record in StreamReader(f, skip_magic=False).records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    channel = channels.get(record.channel_id)
                    if channel is None:
                        continue
                    t = record.log_time / 1e9
                    if channel.topic == TOPIC_IMAGE:
                        if image_decoder is None:
                            image_decoder = decoders.decoder_for(
                                channel.message_encoding, schemas.get(channel.schema_id)
                            )
                        if image_decoder is not None:
                            frames.append((t, bytes(image_decoder(record.data).data)))
                    elif channel.topic in {TOPIC_GYRO, TOPIC_ACCEL}:
                        d = json.loads(record.data)
                        row = (t, float(d["x"]), float(d["y"]), float(d["z"]))
                        (gyro if channel.topic == TOPIC_GYRO else accel).append(row)
                    elif channel.topic == TOPIC_META:
                        meta = json.loads(record.data)
        except (McapError, EOFError, ValueError, struct.error) as exc:
            truncated = True
            logger.warning(
                "recording ends mid-record; using what was read", extra={"error": str(exc)}
            )

    if not frames:
        msg = f"{path} contains no camera frames"
        raise ValueError(msg)
    frames.sort(key=lambda item: item[0])
    return Recording(
        path=path,
        frame_times=np.array([t for t, _ in frames], dtype=np.float64),
        frame_jpegs=[data for _, data in frames],
        gyro=_series(gyro),
        accel=_series(accel),
        meta=meta,
        truncated=truncated,
    )
