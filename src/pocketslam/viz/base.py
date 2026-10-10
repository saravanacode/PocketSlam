"""Visualiser interface and a file-backed trajectory writer."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pocketslam.core.types import DepthMap, Frame, Pose


@runtime_checkable
class Visualizer(Protocol):
    """Receives every processed frame for display or storage."""

    def show(self, frame: Frame, depth: DepthMap, pose: Pose | None) -> None:
        """Handle one processed frame."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


class TrajectoryWriter:
    """Writes tracked poses to a TUM-format trajectory file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("w", encoding="utf-8")

    def show(self, frame: Frame, depth: DepthMap, pose: Pose | None) -> None:
        """Append ``pose`` to the trajectory file, skipping lost frames."""
        if pose is not None:
            self._file.write(pose.to_tum() + "\n")

    def close(self) -> None:
        """Close the trajectory file."""
        self._file.close()
