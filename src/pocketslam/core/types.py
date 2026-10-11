"""Data types shared by every pipeline stage.

This module has no dependencies inside ``pocketslam`` and no torch import, so
any stage can use it without pulling in the others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Pinhole camera intrinsics in pixels.

    Attributes:
        fx: Focal length along x.
        fy: Focal length along y.
        cx: Principal point x.
        cy: Principal point y.
        width: Image width the intrinsics belong to.
        height: Image height the intrinsics belong to.
        source: Where the values came from: ``"device"``, ``"calibration"``
            or ``"fov-estimate"``. Logged so results stay traceable.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> Intrinsics:
        """Estimate intrinsics from a horizontal field of view.

        Use this only when the phone reports no intrinsics and no calibration
        file exists. Square pixels and a centred principal point are assumed.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            hfov_deg: Horizontal field of view in degrees, in (0, 180).

        Returns:
            Estimated intrinsics with ``source="fov-estimate"``.

        Raises:
            ValueError: If the field of view or image size is out of range.
        """
        if not 0.0 < hfov_deg < 180.0:  # noqa: PLR2004 - geometric bound, not a magic number
            msg = f"hfov_deg must be in (0, 180), got {hfov_deg}"
            raise ValueError(msg)
        if width <= 0 or height <= 0:
            msg = f"image size must be positive, got {width}x{height}"
            raise ValueError(msg)
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(
            fx=f,
            fy=f,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
            source="fov-estimate",
        )

    def scaled(self, width: int, height: int) -> Intrinsics:
        """Return intrinsics for the same camera at a different resolution.

        WebRTC senders change resolution on the fly when bandwidth drops, so
        intrinsics must follow the size of each received frame.

        Args:
            width: New image width in pixels.
            height: New image height in pixels.

        Returns:
            Rescaled intrinsics, or ``self`` if the size is unchanged.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx = width / self.width
        sy = height / self.height
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
            source=self.source,
        )

    def matrix(self) -> NDArray[np.float64]:
        """Return the 3x3 camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """What a phone reports about itself when a session starts.

    Attributes:
        width: Actual capture width the browser delivers.
        height: Actual capture height the browser delivers.
        device_label: Free-text device description, for logs only.
        intrinsics: Camera intrinsics if the phone knows them.
        model: Phone model name if the browser reports it (calibration key).
        orientation_deg: Screen rotation (0, 90, 180 or 270) when streaming
            started; tells offline tools how camera axes map to IMU axes.
    """

    width: int
    height: int
    device_label: str = "unknown"
    intrinsics: Intrinsics | None = None
    orientation_deg: int = 0
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ImuSample:
    """One reading from a phone motion sensor, in the phone's device frame.

    Attributes:
        kind: ``"gyro"`` (angular velocity, rad/s) or ``"accel"``
            (acceleration including gravity, m/s^2).
        timestamp_s: Reading time in seconds on the phone clock (Unix epoch).
        x: X-axis value.
        y: Y-axis value.
        z: Z-axis value.
    """

    kind: str
    timestamp_s: float
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class RecordRequest:
    """The phone asks to start (``on=True``) or stop recording a clip."""

    on: bool


@dataclass(frozen=True, slots=True)
class ClipInfo:
    """A finished clip recording.

    Attributes:
        path: The MCAP file.
        frames: Camera frames written.
        imu_samples: IMU samples written.
        seconds: Time between the first and last frame.
    """

    path: Path
    frames: int
    imu_samples: int
    seconds: float


@dataclass(frozen=True, slots=True)
class LinkStats:
    """The phone's own view of its camera and WebRTC sender, sent once a second.

    Every field is optional because browsers report different subsets.

    Attributes:
        camera_fps: Frames per second the camera delivers to the browser.
        sent_fps: Frames per second the encoder sends.
        width: Encoded frame width.
        height: Encoded frame height.
        quality_limitation: Why the encoder lowers quality or frame rate:
            ``"none"``, ``"bandwidth"``, ``"cpu"`` or ``"other"``.
        encoder: Encoder implementation name, e.g. a hardware codec.
        codec: Negotiated codec MIME type.
        encode_ms: Average encode time per frame.
        target_kbps: Bitrate the encoder aims for.
        available_kbps: Outgoing bandwidth the browser's estimator allows.
        rtt_ms: Round-trip time to the laptop.
        nack: Retransmission requests received so far.
        pli: Keyframe requests received so far.
    """

    camera_fps: float | None = None
    sent_fps: float | None = None
    width: int | None = None
    height: int | None = None
    quality_limitation: str | None = None
    encoder: str | None = None
    codec: str | None = None
    encode_ms: float | None = None
    target_kbps: float | None = None
    available_kbps: float | None = None
    rtt_ms: float | None = None
    nack: int | None = None
    pli: int | None = None


@dataclass(frozen=True, slots=True)
class Frame:
    """One RGB camera frame.

    Attributes:
        image: ``(H, W, 3)`` uint8 RGB image.
        timestamp_s: Capture time in seconds on the sender clock.
        index: Frame number within the session, starting at 0.
        intrinsics: Camera intrinsics for this frame.
    """

    image: NDArray[np.uint8]
    timestamp_s: float
    index: int
    intrinsics: Intrinsics

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.image.shape[1])


@dataclass(frozen=True, slots=True)
class DepthMap:
    """Per-pixel depth for one frame.

    Attributes:
        depth: ``(H, W)`` float32 depth. Metres if ``is_metric``, otherwise
            affine-invariant relative depth.
        is_metric: True only when values are in metres. SLAM must never treat
            relative depth as metric.
        timestamp_s: Timestamp of the source frame.
    """

    depth: NDArray[np.float32]
    is_metric: bool
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class MultiViewGeometry:
    """Depth and cameras for a set of views, estimated jointly.

    All arrays share the model's processing resolution ``(H, W)``.

    Attributes:
        depth: ``(N, H, W)`` float32 depth along the camera z axis.
        confidence: ``(N, H, W)`` float32, higher is more reliable.
        intrinsics: ``(N, 3, 3)`` float64 camera matrices at ``(H, W)``.
        world_to_camera: ``(N, 4, 4)`` float64 poses (OpenCV camera axes).
        images: ``(N, H, W, 3)`` uint8 RGB images the model saw.
        is_metric: True only when depth and translations are in metres.
    """

    depth: NDArray[np.float32]
    confidence: NDArray[np.float32]
    intrinsics: NDArray[np.float64]
    world_to_camera: NDArray[np.float64]
    images: NDArray[np.uint8]
    is_metric: bool

    def __len__(self) -> int:
        return int(self.depth.shape[0])

    def camera_to_world(self) -> NDArray[np.float64]:
        """``(N, 4, 4)`` camera-to-world poses."""
        return np.asarray(np.linalg.inv(self.world_to_camera), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Pose:
    """Camera-to-world rigid transform.

    Attributes:
        matrix: ``(4, 4)`` float64 homogeneous transform.
        timestamp_s: Timestamp of the frame this pose belongs to.
    """

    timestamp_s: float
    matrix: NDArray[np.float64] = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    def to_tum(self) -> str:
        """Format the pose as one line of a TUM trajectory file.

        Returns:
            ``"timestamp tx ty tz qx qy qz qw"``.
        """
        r = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        qx, qy, qz, qw = rotation_to_quaternion(r)
        return (
            f"{self.timestamp_s:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )


def rotation_to_quaternion(r: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to a unit quaternion ``(x, y, z, w)``."""
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            float(r[2, 1] - r[1, 2]) / s,
            float(r[0, 2] - r[2, 0]) / s,
            float(r[1, 0] - r[0, 1]) / s,
            0.25 * s,
        )
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        return (
            0.25 * s,
            float(r[0, 1] + r[1, 0]) / s,
            float(r[0, 2] + r[2, 0]) / s,
            float(r[2, 1] - r[1, 2]) / s,
        )
    if r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        return (
            float(r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            float(r[1, 2] + r[2, 1]) / s,
            float(r[0, 2] - r[2, 0]) / s,
        )
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
    return (
        float(r[0, 2] + r[2, 0]) / s,
        float(r[1, 2] + r[2, 1]) / s,
        0.25 * s,
        float(r[1, 0] - r[0, 1]) / s,
    )
