"""Live distance from the phone to the marker.

With the camera's focal length measured (``calib.live``) and the marker's size
known, the marker's four corners in one frame give its full pose by
perspective-n-point. The distance is the length of the camera-to-marker
translation: plain geometry, no depth model, so it runs on every frame.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pocketslam.calib.marker import Detector, make_detector

logger = logging.getLogger(__name__)

SMOOTH_FRAMES = 5
MARKER_FRESH_S = 0.5
UPDATE_PERIOD_S = 1 / 15
HISTORY_SECONDS = 30.0
HISTORY_PERIOD_S = 0.2
CORNER_NOISE_PX = 1.0
RATE_WINDOW_S = 1.0
MAX_REPROJECTION_PX = 5.0

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarkerPose:
    """Where the marker is relative to the camera.

    Attributes:
        distance_m: Camera centre to marker centre.
        tilt_deg: Angle between the viewing ray and the marker's normal
            (0 = seen head-on).
        side_px: Mean edge length of the marker in the image.
    """

    distance_m: float
    tilt_deg: float
    side_px: float


def marker_pose(
    corners: Array, focal_px: float, image_size: tuple[int, int], size_m: float
) -> MarkerPose | None:
    """Pose of a square marker from its four corners (perspective-n-point).

    Args:
        corners: ``(4, 2)`` corners in the ``calib.marker`` pixel convention,
            in the detector's order.
        focal_px: Focal length in pixels at this image size (square pixels,
            principal point at the image centre).
        image_size: ``(width, height)``.
        size_m: Edge of the marker's black square in metres.

    Returns:
        The pose, or None if the solver fails.
    """
    import cv2  # noqa: PLC0415 - optional extra

    half = size_m / 2
    square = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64
    )
    width, height = image_size
    k = np.array([[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]], np.float64)
    pixels = (corners - 0.5).astype(np.float64)
    # The iterative solver, not SOLVEPNP_IPPE_SQUARE: OpenCV 5.0's IPPE returns
    # poses that do not reproject onto the corners for this input.
    ok, rvec, tvec = cv2.solvePnP(square, pixels, k, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(square, rvec, tvec, k, np.zeros(5))
    if float(np.abs(np.asarray(projected).reshape(4, 2) - pixels).max()) > MAX_REPROJECTION_PX:
        return None
    t = np.asarray(tvec, dtype=np.float64).ravel()
    distance = float(np.linalg.norm(t))
    if distance <= 0:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    cos_tilt = abs(float(np.asarray(rotation)[:, 2] @ (t / distance)))
    side = float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
    return MarkerPose(distance, math.degrees(math.acos(min(1.0, cos_tilt))), side)


def distance_error(distance_m: float, side_px: float, focal_rel_std: float) -> float:
    """Rough one-sigma error of a distance reading.

    Combines the focal length's uncertainty with about one pixel of corner
    noise over the marker's size in the image, so it grows with distance.
    """
    pixel_term = CORNER_NOISE_PX / max(side_px, 1.0)
    return distance_m * math.hypot(focal_rel_std, pixel_term)


@dataclass
class _State:
    active: bool = False
    device: str = "unknown"
    focal_ratio: float = 0.0
    focal_rel_std: float = 0.0
    image_size: tuple[int, int] = (0, 0)
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTH_FRAMES))
    distance_m: float | None = None
    raw_m: float | None = None
    error_m: float | None = None
    tilt_deg: float | None = None
    corners: Array | None = None
    last_seen: float = 0.0
    min_m: float | None = None
    max_m: float | None = None
    history: deque[tuple[float, float]] = field(default_factory=deque)
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    started: float = 0.0
    error: str = ""


class LiveDistance:
    """Distance to the marker from live frames. Thread-safe."""

    def __init__(
        self,
        marker_size_m: float | None,
        marker_id: int | None,
        dictionary: str,
        detector: Detector | None = None,
    ) -> None:
        self._size_m = marker_size_m
        self._marker_id = marker_id
        self._dictionary = dictionary
        self._detector = detector
        self._state = _State()
        self._lock = threading.Lock()
        self._frame: NDArray[np.uint8] | None = None
        self._frame_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_update = 0.0
        # Called from the worker thread with a fresh snapshot, about 15 times a second.
        self.on_update: Callable[[dict[str, Any]], None] | None = None

    def start(self, device: str, focal_ratio: float, focal_rel_std: float) -> None:
        """Begin measuring for ``device`` with its calibrated focal length.

        Args:
            device: Phone label, for display.
            focal_ratio: Focal length divided by the longer image side.
            focal_rel_std: Relative uncertainty of that focal length.

        Raises:
            RuntimeError: If the marker size is not configured or OpenCV is missing.
        """
        if self._size_m is None:
            msg = (
                "Marker size is not set. Add POCKETSLAM_RECON_MARKER_SIZE_M (metres) "
                "to config/.env and restart the server."
            )
            raise RuntimeError(msg)
        if self._detector is None:
            self._detector = make_detector(self._dictionary, self._marker_id)
        self.stop()
        with self._lock:
            self._state = _State(
                active=True,
                device=device,
                focal_ratio=focal_ratio,
                focal_rel_std=focal_rel_std,
                started=time.monotonic(),
            )
        self._thread = threading.Thread(target=self._run, name="live-distance", daemon=True)
        self._thread.start()
        logger.info("live distance started", extra={"device": device, "focal_ratio": focal_ratio})

    def stop(self) -> None:
        """Stop measuring."""
        with self._lock:
            was_active = self._state.active
            self._state.active = False
        self._frame_ready.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        if was_active and self.on_update is not None:
            self.on_update(self.snapshot())

    @property
    def active(self) -> bool:
        """True while measuring."""
        return self._state.active

    def submit(self, image: NDArray[np.uint8], _t_s: float = 0.0) -> None:
        """Offer a live frame. Only the newest is kept; cheap when inactive."""
        if self._state.active:
            self._frame = image
            self._frame_ready.set()

    def _run(self) -> None:
        while self._state.active:
            self._frame_ready.wait(timeout=0.5)
            self._frame_ready.clear()
            frame, self._frame = self._frame, None
            if frame is None or not self._state.active:
                continue
            try:
                self._process(frame)
            except Exception as exc:  # report to the user instead of dying silently
                logger.exception("live distance failed")
                with self._lock:
                    self._state.error = str(exc)
                    self._state.active = False
            now = time.monotonic()
            done = not self._state.active
            if self.on_update is not None and (done or now - self._last_update >= UPDATE_PERIOD_S):
                self._last_update = now
                self.on_update(self.snapshot())

    def _process(self, frame: NDArray[np.uint8]) -> None:
        assert self._detector is not None  # noqa: S101 - set in start()
        assert self._size_m is not None  # noqa: S101 - checked in start()
        height, width = frame.shape[:2]
        now = time.monotonic()
        found = self._detector(frame)
        pose = None
        if found is not None:
            focal_px = self._state.focal_ratio * max(width, height)
            pose = marker_pose(found[1], focal_px, (width, height), self._size_m)
        with self._lock:
            st = self._state
            st.image_size = (width, height)
            st.frame_times.append(now)
            if found is None or pose is None:
                return
            st.corners, st.last_seen = found[1], now
            st.recent.append(pose.distance_m)
            st.raw_m = pose.distance_m
            st.distance_m = float(np.median(st.recent))
            st.tilt_deg = pose.tilt_deg
            st.error_m = distance_error(st.distance_m, pose.side_px, st.focal_rel_std)
            st.min_m = st.distance_m if st.min_m is None else min(st.min_m, st.distance_m)
            st.max_m = st.distance_m if st.max_m is None else max(st.max_m, st.distance_m)
            t = now - st.started
            if not st.history or t - st.history[-1][0] >= HISTORY_PERIOD_S:
                st.history.append((t, st.distance_m))
            while st.history and t - st.history[0][0] > HISTORY_SECONDS:
                st.history.popleft()

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready state for the dashboard and the phone."""
        with self._lock:
            st = self._state
            now = time.monotonic()
            visible = st.corners is not None and now - st.last_seen < MARKER_FRESH_S
            width, height = st.image_size
            corners = None
            if visible and st.corners is not None and width and height:
                corners = (st.corners / np.array([width, height])).round(4).tolist()
            rate = sum(1 for t in st.frame_times if now - t <= RATE_WINDOW_S) / RATE_WINDOW_S

            def r(value: float | None, digits: int = 4) -> float | None:
                return None if value is None else round(value, digits)

            return {
                "active": st.active,
                "error": st.error,
                "device": st.device,
                "marker_visible": visible,
                "distance_m": r(st.distance_m),
                "raw_m": r(st.raw_m),
                "error_m": r(st.error_m),
                "tilt_deg": r(st.tilt_deg, 1),
                "min_m": r(st.min_m),
                "max_m": r(st.max_m),
                "corners": corners,
                "rate_hz": round(rate, 1),
                "focal_px": round(st.focal_ratio * max(width, height), 1) if width else None,
                "history": [[round(t, 2), round(d, 4)] for t, d in st.history],
            }
