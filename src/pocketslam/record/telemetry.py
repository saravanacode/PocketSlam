"""Foxglove channels for phone data: one context feeds MCAP files and the live viewer.

Every message logged here goes to all sinks attached to the context: the
WebSocket server (live view in Foxglove) and the MCAP writer of the current
recording, if one is open.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import foxglove
from foxglove.channels import CameraCalibrationChannel, CompressedImageChannel
from foxglove.messages import CameraCalibration, CompressedImage, Timestamp
from foxglove.websocket import WebSocketServer

from pocketslam.core.topics import (
    TOPIC_ACCEL,
    TOPIC_CALIBRATION,
    TOPIC_GYRO,
    TOPIC_IMAGE,
    TOPIC_META,
)
from pocketslam.core.types import ImuSample, Intrinsics, SessionMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "TOPIC_ACCEL",
    "TOPIC_CALIBRATION",
    "TOPIC_GYRO",
    "TOPIC_IMAGE",
    "TOPIC_META",
    "Telemetry",
]

CAMERA_FRAME = "camera"
IMU_FRAME = "imu"

_TIMESTAMP_SCHEMA = {
    "type": "object",
    "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
}
IMU_SCHEMA: dict[str, Any] = {
    "title": "pocketslam.ImuVector",
    "type": "object",
    "properties": {
        "timestamp": _TIMESTAMP_SCHEMA,
        "frame_id": {"type": "string"},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
}
META_SCHEMA: dict[str, Any] = {
    "title": "pocketslam.SessionMetadata",
    "type": "object",
    "properties": {
        "timestamp": _TIMESTAMP_SCHEMA,
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "device": {"type": "string"},
        "intrinsics_source": {"type": "string"},
        "orientation": {"type": "integer"},
        "model": {"type": "string"},
    },
}


def to_ns(t_s: float) -> int:
    """Convert seconds to integer nanoseconds."""
    return round(t_s * 1e9)


def _stamp(t_s: float) -> dict[str, int]:
    ns = to_ns(t_s)
    return {"sec": ns // 1_000_000_000, "nsec": ns % 1_000_000_000}


class Telemetry:
    """Owns the Foxglove context, its channels and the live WebSocket server.

    Create one per process. Channels are thread-safe, so a recording worker
    thread may log while the event loop runs.
    """

    def __init__(self) -> None:
        self.context = foxglove.Context()
        self._image = CompressedImageChannel(TOPIC_IMAGE, context=self.context)
        self._calibration = CameraCalibrationChannel(TOPIC_CALIBRATION, context=self.context)
        self._gyro = foxglove.Channel(TOPIC_GYRO, schema=IMU_SCHEMA, context=self.context)
        self._accel = foxglove.Channel(TOPIC_ACCEL, schema=IMU_SCHEMA, context=self.context)
        self._meta = foxglove.Channel(TOPIC_META, schema=META_SCHEMA, context=self.context)
        self._server: WebSocketServer | None = None

    def start_server(self, host: str, port: int) -> int:
        """Start the live-view WebSocket server.

        Args:
            host: Interface to bind.
            port: TCP port; 0 picks a free one.

        Returns:
            The port the server listens on.
        """
        self._server = foxglove.start_server(
            name="PocketSlam", host=host, port=port, context=self.context
        )
        logger.info(
            "live view server started",
            extra={"url": f"ws://{host}:{self._server.port}", "app_url": self._server.app_url()},
        )
        return int(self._server.port)

    def stop_server(self) -> None:
        """Stop the live-view server if it runs."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def open_recording(self, path: Path) -> foxglove.MCAPWriter:
        """Start writing every logged message to a new MCAP file.

        Args:
            path: Output file. Parent directories are created.

        Returns:
            The writer; call ``close()`` to finish the file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        return foxglove.open_mcap(path, context=self.context)

    def log_image(self, jpeg: bytes, t_s: float) -> None:
        """Log one JPEG camera frame captured at host time ``t_s``."""
        msg = CompressedImage(
            timestamp=Timestamp(*divmod(to_ns(t_s), 1_000_000_000)),
            frame_id=CAMERA_FRAME,
            data=jpeg,
            format="jpeg",
        )
        self._image.log(msg, log_time=to_ns(t_s))

    def log_calibration(self, k: Intrinsics, t_s: float) -> None:
        """Log pinhole intrinsics with no distortion model."""
        msg = CameraCalibration(
            timestamp=Timestamp(*divmod(to_ns(t_s), 1_000_000_000)),
            frame_id=CAMERA_FRAME,
            width=k.width,
            height=k.height,
            distortion_model="plumb_bob",
            D=[0.0, 0.0, 0.0, 0.0, 0.0],
            K=[k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0],
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            P=[k.fx, 0.0, k.cx, 0.0, 0.0, k.fy, k.cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self._calibration.log(msg, log_time=to_ns(t_s))

    def log_imu(self, sample: ImuSample, t_s: float) -> None:
        """Log one IMU sample at host time ``t_s``."""
        channel = self._gyro if sample.kind == "gyro" else self._accel
        channel.log(
            {
                "timestamp": _stamp(t_s),
                "frame_id": IMU_FRAME,
                "x": sample.x,
                "y": sample.y,
                "z": sample.z,
            },
            log_time=to_ns(t_s),
        )

    def log_meta(self, meta: SessionMetadata, t_s: float) -> None:
        """Log the phone's session metadata."""
        self._meta.log(
            {
                "timestamp": _stamp(t_s),
                "width": meta.width,
                "height": meta.height,
                "device": meta.device_label,
                "intrinsics_source": meta.intrinsics.source if meta.intrinsics else "none",
                "orientation": meta.orientation_deg,
                "model": meta.model or "",
            },
            log_time=to_ns(t_s),
        )
