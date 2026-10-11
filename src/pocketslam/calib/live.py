"""Live camera calibration: measure the phone's focal length from the live stream.

The user points the phone at the marker and moves around it. A background
thread detects the marker in the newest frame, keeps a varied set of views,
and re-estimates the focal length as views come in. It reports what is still
missing (closer, or a view from one side) and saves the result once the
uncertainty is low enough.

The focal length is a property of the camera, so one calibration serves every
later recording from the same phone and stream resolution.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pocketslam.calib.marker import (
    Detector,
    FocalEstimate,
    estimate_focal,
    make_detector,
)
from pocketslam.calib.store import ScaleCalibration, saved_focal

logger = logging.getLogger(__name__)

SIDES = ("left", "right", "above", "below")
MIN_SIZE_FRACTION = 0.12  # marker edge / longer image side, to accept a view
GOOD_SIZE_FRACTION = 0.2
TILT_RATIO = 1.10  # near edge / far edge that counts as a tilted view
VIEWS_PER_SIDE = 3
MIN_VIEW_SHIFT_PX = 25.0
MAX_VIEWS = 80
MIN_VIEWS_TO_SAVE = 10
ESTIMATE_EVERY_N_VIEWS = 2
MIN_FOV_DEG, MAX_FOV_DEG = 20.0, 120.0
UPDATE_PERIOD_S = 0.1
MARKER_FRESH_S = 0.5

Array = NDArray[np.float64]


def view_side(corners: Array) -> str | None:
    """Which side of the marker the camera is on, from perspective foreshortening.

    The marker edge nearer the camera looks longer. A longer left edge means
    the camera is to the marker's left, and so on.

    Returns:
        ``"left"``, ``"right"``, ``"above"``, ``"below"``, or None for a
        near head-on view.
    """
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    mids = np.array([(a + b) / 2 for a, b in edges])
    lengths = np.array([np.linalg.norm(a - b) for a, b in edges])
    left, right = lengths[np.argmin(mids[:, 0])], lengths[np.argmax(mids[:, 0])]
    top, bottom = lengths[np.argmin(mids[:, 1])], lengths[np.argmax(mids[:, 1])]
    horizontal = max(left, right) / min(left, right)
    vertical = max(top, bottom) / min(top, bottom)
    if max(horizontal, vertical) < TILT_RATIO:
        return None
    if horizontal >= vertical:
        return "left" if left > right else "right"
    return "above" if top > bottom else "below"


@dataclass
class _State:
    active: bool = False
    device: str = "unknown"
    image_size: tuple[int, int] = (0, 0)
    views: list[Array] = field(default_factory=list)
    sides: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SIDES, 0))
    last_corners: Array | None = None
    last_seen: float = 0.0
    size_fraction: float = 0.0
    estimate: FocalEstimate | None = None
    saved: bool = False
    error: str = ""


class LiveCalibration:
    """Focal-length calibration fed by live frames. Thread-safe."""

    def __init__(
        self,
        store: ScaleCalibration,
        marker_size_m: float | None,
        marker_id: int | None,
        dictionary: str,
        detector: Detector | None = None,
    ) -> None:
        self._store = store
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
        # Called from the worker thread with a fresh snapshot, about 10 times a second.
        self.on_update: Callable[[dict[str, Any]], None] | None = None
        # Called from the worker thread once a calibration is saved:
        # (device, focal / longer image side, relative uncertainty).
        self.on_saved: Callable[[str, float, float], None] | None = None

    def start(self, device: str) -> None:
        """Begin a new calibration for ``device``.

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
            self._state = _State(active=True, device=device)
        self._thread = threading.Thread(target=self._run, name="live-calibration", daemon=True)
        self._thread.start()
        logger.info("live calibration started", extra={"device": device})

    def stop(self) -> None:
        """Stop collecting views. A saved result stays saved."""
        with self._lock:
            self._state.active = False
        self._frame_ready.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def marker(self) -> dict[str, Any]:
        """The configured marker: dictionary, id (None for any) and size in metres."""
        return {"dictionary": self._dictionary, "id": self._marker_id, "size_m": self._size_m}

    def saved_focal_ratio(self, device: str) -> float | None:
        """Stored ``focal / longer image side`` for ``device``, if it was calibrated."""
        stored = self._store.focal_ratio(device)
        return None if stored is None else stored[0]

    def saved_focal(self, device: str) -> tuple[float, float] | None:
        """Stored ``(focal ratio, relative uncertainty)`` for ``device``, if calibrated."""
        return saved_focal(self._store, device)

    @property
    def active(self) -> bool:
        """True while views are being collected."""
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
                logger.exception("live calibration failed")
                with self._lock:
                    self._state.error = str(exc)
                    self._state.active = False
            self._notify()

    def _process(self, frame: NDArray[np.uint8]) -> None:
        assert self._detector is not None  # noqa: S101 - set in start()
        assert self._size_m is not None  # noqa: S101 - checked in start()
        height, width = frame.shape[:2]
        found = self._detector(frame)
        with self._lock:
            st = self._state
            st.image_size = (width, height)
            if found is None:
                return
            corners = found[1]
            st.last_corners, st.last_seen = corners, time.monotonic()
            side_px = float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
            st.size_fraction = side_px / max(width, height)
            if st.size_fraction < MIN_SIZE_FRACTION or len(st.views) >= MAX_VIEWS:
                return
            if (
                st.views
                and min(float(np.mean(np.linalg.norm(corners - v, axis=1))) for v in st.views)
                < MIN_VIEW_SHIFT_PX
            ):
                return  # too similar to a view we already have
            st.views.append(corners)
            side = view_side(corners)
            if side is not None:
                st.sides[side] += 1
            due = len(st.views) >= MIN_VIEWS_TO_SAVE and len(st.views) % ESTIMATE_EVERY_N_VIEWS == 0
            views = list(st.views) if due else None
        if views is None:
            return
        estimate = estimate_focal(views, (width, height), self._size_m)
        saved: tuple[str, float, float] | None = None
        with self._lock:
            st = self._state
            st.estimate = estimate
            if estimate is not None and self._ready_to_save(st, estimate):
                ratio = estimate.focal_px / max(width, height)
                self._store.add_focal(st.device, ratio, estimate.rel_std, "live")
                st.saved, st.active = True, False
                saved = (st.device, ratio, estimate.rel_std)
                logger.info(
                    "live calibration saved",
                    extra={
                        "device": st.device,
                        "focal_px": round(estimate.focal_px),
                        "rel_std": round(estimate.rel_std, 4),
                        "views": len(st.views),
                    },
                )
        if saved is not None and self.on_saved is not None:
            self.on_saved(*saved)

    @staticmethod
    def _ready_to_save(st: _State, estimate: FocalEstimate) -> bool:
        fov = _fov_deg(estimate.focal_px, st.image_size)
        both_axes = max(st.sides["left"], st.sides["right"]) >= VIEWS_PER_SIDE and (
            max(st.sides["above"], st.sides["below"]) >= VIEWS_PER_SIDE
        )
        return estimate.reliable and both_axes and MIN_FOV_DEG <= fov <= MAX_FOV_DEG

    def _notify(self) -> None:
        now = time.monotonic()
        done = not self._state.active
        if self.on_update is not None and (done or now - self._last_update >= UPDATE_PERIOD_S):
            self._last_update = now
            self.on_update(self.snapshot())

    def _hint(self, st: _State, visible: bool) -> str:
        if st.error:
            return st.error
        if st.saved:
            return "Saved. Recordings from this phone now use the measured focal length."
        return self._guidance(st, visible) if st.active else ""

    @staticmethod
    def _guidance(st: _State, visible: bool) -> str:
        """What the user should do next while collecting views."""
        if not visible:
            return "Point the phone at the marker."
        if st.size_fraction < GOOD_SIZE_FRACTION:
            return "Move closer: the marker should fill about a third of the picture."
        missing = next((side for side in SIDES if st.sides[side] < VIEWS_PER_SIDE), None)
        if missing is not None:
            where = {"left": "to the left of", "right": "to the right of"}.get(missing, missing)
            return f"Move the phone {where} the marker and aim back at it."
        if st.estimate is not None and not st.estimate.reliable:
            return "Almost there. Keep circling the marker at steep angles."
        return "Keep moving around the marker."

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready state for the dashboard."""
        with self._lock:
            st = self._state
            visible = st.last_corners is not None and (
                time.monotonic() - st.last_seen < MARKER_FRESH_S
            )
            width, height = st.image_size
            corners = None
            if visible and st.last_corners is not None and width and height:
                corners = (st.last_corners / np.array([width, height])).round(4).tolist()
            est = st.estimate
            return {
                "active": st.active,
                "saved": st.saved,
                "error": st.error,
                "device": st.device,
                "marker_visible": visible,
                "corners": corners,
                "size_fraction": round(st.size_fraction, 3),
                "views": len(st.views),
                "views_needed": MIN_VIEWS_TO_SAVE,
                "sides": dict(st.sides),
                "sides_needed": VIEWS_PER_SIDE,
                "focal_px": None if est is None else round(est.focal_px, 1),
                "fov_deg": None if est is None else round(_fov_deg(est.focal_px, st.image_size), 1),
                "rel_std": None if est is None else round(est.rel_std, 4),
                "image_size": [width, height],
                "hint": self._hint(st, visible),
            }


def _fov_deg(focal_px: float, image_size: tuple[int, int]) -> float:
    """Field of view along the longer image side, in degrees."""
    return 2 * math.degrees(math.atan(max(image_size) / 2 / focal_px)) if focal_px > 0 else 0.0
