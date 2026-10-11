"""Remembered scale calibration, learned from scans that showed the marker.

The metric depth model's scale is biased in a way that is fairly stable for
one phone camera. Each scan with a marker tells us that bias:
``ratio = marker scale / metric-depth scale``. The median ratio is applied to
later scans of the same phone that show no marker.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_SAMPLES = 20


class ScaleCalibration:
    """JSON-backed store of marker-to-metric-depth ratios per phone."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"devices": {}}
        except (OSError, ValueError):
            logger.warning(
                "calibration file unreadable; ignoring it", extra={"path": str(self.path)}
            )
            return {"devices": {}}
        if not isinstance(data, dict) or not isinstance(data.get("devices"), dict):
            return {"devices": {}}
        return data

    def add_ratio(self, device: str, ratio: float, scan: str) -> None:
        """Record one ``marker / metric-depth`` ratio for ``device``."""
        data = self._load()
        samples = data["devices"].setdefault(device, {}).setdefault("metric_ratio", [])
        samples.append({"ratio": round(ratio, 5), "scan": scan, "time": round(time.time())})
        del samples[:-MAX_SAMPLES]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def ratio(self, device: str) -> tuple[float, int, bool] | None:
        """Correction factor for ``device``.

        Returns:
            ``(factor, sample count, same_device)``. Samples from other phones
            are used only when this phone has none. None if nothing is stored.
        """
        devices = self._load()["devices"]
        own = [s["ratio"] for s in devices.get(device, {}).get("metric_ratio", [])]
        if own:
            return float(statistics.median(own)), len(own), True
        others = [s["ratio"] for d in devices.values() for s in d.get("metric_ratio", [])]
        if others:
            return float(statistics.median(others)), len(others), False
        return None

    def add_focal(self, device: str, focal_ratio: float, rel_std: float, scan: str) -> None:
        """Record a measured focal length for ``device``.

        Args:
            device: Phone label.
            focal_ratio: Focal length divided by the image's longer side, which
                makes it independent of the stream resolution.
            rel_std: Relative uncertainty of the measurement.
            scan: Name of the clip it came from.
        """
        data = self._load()
        samples = data["devices"].setdefault(device, {}).setdefault("focal_ratio", [])
        samples.append(
            {
                "ratio": round(focal_ratio, 5),
                "rel_std": round(rel_std, 4),
                "scan": scan,
                "time": round(time.time()),
            }
        )
        del samples[:-MAX_SAMPLES]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def focal_samples(self, device: str) -> list[dict[str, Any]]:
        """Stored focal-length measurements for ``device``, oldest first."""
        return list(self._load()["devices"].get(device, {}).get("focal_ratio", []))

    def focal_ratio(self, device: str) -> tuple[float, int] | None:
        """Median measured ``focal / longer image side`` for ``device``, with sample count."""
        samples = self._load()["devices"].get(device, {}).get("focal_ratio", [])
        if not samples:
            return None
        return float(statistics.median(s["ratio"] for s in samples)), len(samples)


def saved_focal(store: ScaleCalibration, device: str) -> tuple[float, float] | None:
    """Measured ``(focal / longer image side, relative uncertainty)`` for ``device``.

    Both are medians over the stored calibrations. None if the phone was never calibrated.
    """
    samples = store.focal_samples(device)
    if not samples:
        return None
    return (
        float(statistics.median(s["ratio"] for s in samples)),
        float(statistics.median(s.get("rel_std", 0.0) for s in samples)),
    )


def device_label(meta: dict[str, Any] | None) -> str:
    """Key for per-phone calibration: the phone model if known, else its user agent."""
    meta = meta or {}
    return str(meta.get("model") or meta.get("device") or "unknown")
