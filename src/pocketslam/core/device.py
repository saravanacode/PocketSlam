"""Compute device selection.

This is the only module allowed to name a backend. ``scripts/check_structure.py``
fails the build if backend names appear anywhere else in ``src/``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_BACKENDS = ("cuda", "mps", "cpu")


def available_backends() -> list[str]:
    """Return the backends usable on this machine, fastest first."""
    found: list[str] = []
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    found.append("cpu")
    return found


def select_device(preference: str | None = None) -> torch.device:
    """Pick the compute device.

    Order without a preference: cuda, then mps, then cpu.

    Args:
        preference: ``"cuda"``, ``"mps"``, ``"cpu"``, or None for automatic.
            Usually comes from the ``POCKETSLAM_DEVICE`` setting.

    Returns:
        The selected torch device.

    Raises:
        ValueError: If the preference is not a known backend.
        RuntimeError: If the preferred backend is not available here.
    """
    available = available_backends()
    if preference is None:
        choice = available[0]
    else:
        choice = preference.lower()
        if choice not in _BACKENDS:
            msg = f"unknown device {preference!r}, expected one of {_BACKENDS}"
            raise ValueError(msg)
        if choice not in available:
            msg = f"device {choice!r} requested but only {available} are available"
            raise RuntimeError(msg)
    logger.info("compute device selected", extra={"device": choice, "available": available})
    return torch.device(choice)
