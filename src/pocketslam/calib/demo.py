"""Distance demo: calibrate on the marker if needed, then show live distance to it.

A small state machine over the two live tools:

* ``calibrating``: ``LiveCalibration`` measures the focal length. Entered only
  when the phone has no saved calibration, or the user asked to calibrate again.
* ``measuring``: ``LiveDistance`` reports the distance on every frame. Entered
  directly for a calibrated phone, or automatically when calibration saves.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pocketslam.calib.distance import LiveDistance
from pocketslam.calib.live import LiveCalibration

logger = logging.getLogger(__name__)


class DistanceDemo:
    """Runs the calibrate-then-measure demo. Thread-safe."""

    def __init__(self, calibration: LiveCalibration, distance: LiveDistance) -> None:
        self.calibration = calibration
        self.distance = distance
        self._lock = threading.Lock()
        self._pending = False  # a demo is waiting for calibration to finish
        calibration.on_saved = self._calibrated

    @property
    def stage(self) -> str:
        """``"idle"``, ``"calibrating"`` or ``"measuring"``."""
        if self.distance.active:
            return "measuring"
        if self._pending and self.calibration.active:
            return "calibrating"
        return "idle"

    def start(self, device: str, recalibrate: bool = False) -> str:
        """Start the demo for ``device`` and return the stage it entered.

        Args:
            device: Phone label used to look up its saved calibration.
            recalibrate: Measure the focal length again even if one is saved.

        Raises:
            RuntimeError: If the marker size is not configured or OpenCV is missing.
        """
        self.stop()
        saved = None if recalibrate else self.calibration.saved_focal(device)
        if saved is not None:
            self.distance.start(device, *saved)
        else:
            with self._lock:
                self._pending = True
            try:
                self.calibration.start(device)
            except RuntimeError:
                with self._lock:
                    self._pending = False
                raise
        logger.info("distance demo started", extra={"device": device, "stage": self.stage})
        return self.stage

    def _calibrated(self, device: str, focal_ratio: float, rel_std: float) -> None:
        with self._lock:
            pending, self._pending = self._pending, False
        if pending:
            self.distance.start(device, focal_ratio, rel_std)

    def stop(self) -> None:
        """Stop whichever stage is running."""
        with self._lock:
            pending, self._pending = self._pending, False
        if pending:
            self.calibration.stop()
        self.distance.stop()

    def submit(self, image: NDArray[np.uint8], t_s: float = 0.0) -> None:
        """Offer a live frame to whichever tool is active."""
        self.calibration.submit(image, t_s)
        self.distance.submit(image, t_s)

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready state: the stage plus the distance readout."""
        return {"stage": self.stage, "distance": self.distance.snapshot()}
