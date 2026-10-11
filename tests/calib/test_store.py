from pathlib import Path

import pytest

from pocketslam.calib.store import MAX_SAMPLES, ScaleCalibration


def test_median_ratio_per_device(tmp_path: Path) -> None:
    cal = ScaleCalibration(tmp_path / "cal" / "scale.json")
    assert cal.ratio("phone-a") is None
    for r in (0.80, 0.90, 0.84):
        cal.add_ratio("phone-a", r, "scan")
    assert cal.ratio("phone-a") == (pytest.approx(0.84), 3, True)


def test_falls_back_to_other_devices(tmp_path: Path) -> None:
    cal = ScaleCalibration(tmp_path / "scale.json")
    cal.add_ratio("phone-a", 0.8, "scan")
    assert cal.ratio("phone-b") == (pytest.approx(0.8), 1, False)


def test_keeps_only_recent_samples(tmp_path: Path) -> None:
    cal = ScaleCalibration(tmp_path / "scale.json")
    for i in range(MAX_SAMPLES + 5):
        cal.add_ratio("p", 1.0 + i, "scan")
    factor, count, _ = cal.ratio("p") or (0, 0, False)
    assert count == MAX_SAMPLES
    assert factor > 5  # the oldest (smallest) samples were dropped


def test_corrupt_file_is_ignored_then_replaced(tmp_path: Path) -> None:
    path = tmp_path / "scale.json"
    path.write_text("{not json")
    cal = ScaleCalibration(path)
    assert cal.ratio("p") is None
    cal.add_ratio("p", 0.9, "scan")
    assert cal.ratio("p") == (pytest.approx(0.9), 1, True)


def test_focal_ratio_median(tmp_path: Path) -> None:
    cal = ScaleCalibration(tmp_path / "scale.json")
    assert cal.focal_ratio("p") is None
    for r in (0.66, 0.70, 0.68):
        cal.add_focal("p", r, 0.02, "clip")
    cal.add_ratio("p", 0.8, "scan")  # both kinds of sample live side by side
    assert cal.focal_ratio("p") == (pytest.approx(0.68), 3)
    assert cal.ratio("p") == (pytest.approx(0.8), 1, True)
    assert cal.focal_ratio("other") is None
