import json

import pytest

from pocketslam.transport.webrtc import MetadataError, parse_metadata


def _msg(**fields: object) -> str:
    return json.dumps({"type": "meta", "width": 1280, "height": 720, **fields})


def test_parses_metadata_with_intrinsics() -> None:
    meta = parse_metadata(
        _msg(device="Pixel", intrinsics={"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}), 4096
    )
    assert (meta.width, meta.height, meta.device_label) == (1280, 720, "Pixel")
    assert meta.intrinsics is not None
    assert meta.intrinsics.source == "device"
    assert meta.intrinsics.fx == 1000.0


def test_intrinsics_optional() -> None:
    assert parse_metadata(_msg(), 4096).intrinsics is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps([1, 2]),
        json.dumps({"type": "other", "width": 1, "height": 1}),
        _msg(width=0),
        _msg(width=True),
        _msg(height=100_000),
        _msg(intrinsics={"fx": 1}),
        _msg(intrinsics={"fx": -1, "fy": 1, "cx": 0, "cy": 0}),
        _msg(intrinsics="bad"),
    ],
)
def test_rejects_invalid(raw: str) -> None:
    with pytest.raises(MetadataError):
        parse_metadata(raw, 4096)


def test_rejects_oversized() -> None:
    with pytest.raises(MetadataError, match="limit"):
        parse_metadata(_msg(device="x" * 500), 256)
