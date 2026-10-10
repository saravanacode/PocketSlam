from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pocketslam.config import Settings
from pocketslam.core.types import Frame, Intrinsics


@pytest.fixture
def make_frame() -> Callable[..., Frame]:
    def _make(width: int = 64, height: int = 48, timestamp_s: float = 0.0, index: int = 0) -> Frame:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return Frame(image, timestamp_s, index, Intrinsics.from_fov(width, height, 60.0))

    return _make


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        allow_insecure=True,
        output_dir=tmp_path / "outputs",
        web_dir=Path(__file__).resolve().parents[1] / "web",
        _env_file=None,  # type: ignore[call-arg]
    )
