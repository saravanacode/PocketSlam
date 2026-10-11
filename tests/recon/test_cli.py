from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pocketslam.calib.marker import make_detector

from pocketslam.recon.cli import marker_main, parse_args


def test_parses_options() -> None:
    args = parse_args(
        ["rec.mcap", "--max-frames", "24", "--start", "1.5", "--duration", "5", "--no-metric"]
    )
    assert args.recording == Path("rec.mcap")
    assert (args.max_frames, args.start, args.duration, args.no_metric) == (24, 1.5, 5.0, True)


def test_requires_recording() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_parses_marker_options() -> None:
    args = parse_args(["rec.mcap", "--marker-size", "0.16", "--marker-id", "18"])
    assert (args.marker_size, args.marker_id) == (0.16, 18)


def test_marker_command_writes_image_with_white_border(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    out = tmp_path / "marker.png"
    assert marker_main(["--id", "18", "--side-px", "200", "--out", str(out)]) == 0
    with Image.open(out) as img:
        pixels = np.asarray(img)
    assert pixels.shape == (300, 300)
    assert (pixels[:40] == 255).all()  # top quiet zone
    found = make_detector("DICT_4X4_50", 18)(np.stack([pixels] * 3, axis=-1))
    assert found is not None
