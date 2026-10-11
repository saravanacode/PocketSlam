import asyncio
import json

import pytest

from pocketslam.core.types import ImuSample, LinkStats, SessionMetadata
from pocketslam.transport.webrtc import MessageError, parse_message, skip_to_newest


def _meta(**fields: object) -> str:
    return json.dumps({"type": "meta", "width": 1280, "height": 720, **fields})


def _imu(samples: object) -> str:
    return json.dumps({"type": "imu", "samples": samples})


def test_parses_metadata_with_intrinsics() -> None:
    meta = parse_message(
        _meta(device="Pixel", intrinsics={"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}), 4096
    )
    assert isinstance(meta, SessionMetadata)
    assert (meta.width, meta.height, meta.device_label) == (1280, 720, "Pixel")
    assert meta.intrinsics is not None
    assert meta.intrinsics.source == "device"
    assert meta.intrinsics.fx == 1000.0


def test_intrinsics_optional() -> None:
    meta = parse_message(_meta(), 4096)
    assert isinstance(meta, SessionMetadata)
    assert meta.intrinsics is None


def test_parses_imu_batch_and_converts_ms_to_s() -> None:
    samples = parse_message(_imu([["g", 1500.0, 0.1, 0.2, 0.3], ["a", 1510, 0, 9.8, 0]]), 4096)
    assert samples == [
        ImuSample("gyro", 1.5, 0.1, 0.2, 0.3),
        ImuSample("accel", 1.51, 0.0, 9.8, 0.0),
    ]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps([1, 2]),
        json.dumps({"type": "other", "width": 1, "height": 1}),
        _meta(width=0),
        _meta(width=True),
        _meta(height=100_000),
        _meta(intrinsics={"fx": 1}),
        _meta(intrinsics={"fx": -1, "fy": 1, "cx": 0, "cy": 0}),
        _meta(intrinsics="bad"),
        _meta(orientation=45),
        _imu("bad"),
        _imu([["x", 1, 0, 0, 0]]),
        _imu([["g", 1, 0, 0]]),
        _imu([["g", "1", 0, 0, 0]]),
        _imu([["g", 1, True, 0, 0]]),
        _imu([["g", 1, 0, 0, 0]] * 501),
        json.dumps({"type": "stats", "sent_fps": "fast"}),
        json.dumps({"type": "stats", "codec": 8}),
        json.dumps({"type": "stats", "nack": True}),
    ],
)
def test_rejects_invalid(raw: str) -> None:
    with pytest.raises(MessageError):
        parse_message(raw, 100_000)


def test_rejects_non_finite_imu() -> None:
    raw = '{"type": "imu", "samples": [["g", 1, NaN, 0, 0]]}'
    with pytest.raises(MessageError, match="finite"):
        parse_message(raw, 4096)


def test_rejects_oversized() -> None:
    with pytest.raises(MessageError, match="limit"):
        parse_message(_meta(device="x" * 500), 256)


def test_parses_link_stats_and_ignores_unknown_fields() -> None:
    raw = json.dumps(
        {
            "type": "stats",
            "camera_fps": 30,
            "sent_fps": 4.5,
            "quality_limitation": "bandwidth",
            "pli": 2.0,
            "something_new": [1, 2],
        }
    )
    assert parse_message(raw, 4096) == LinkStats(
        camera_fps=30.0, sent_fps=4.5, quality_limitation="bandwidth", pli=2
    )


def test_parses_orientation() -> None:
    meta = parse_message(_meta(orientation=90), 4096)
    assert isinstance(meta, SessionMetadata)
    assert meta.orientation_deg == 90


def test_parses_phone_model() -> None:
    meta = parse_message(_meta(model="CPH2653"), 4096)
    assert isinstance(meta, SessionMetadata)
    assert meta.model == "CPH2653"
    assert parse_message(_meta(), 4096).model is None  # type: ignore[union-attr]


class FakeTrack:
    def __init__(self, items: list[object]) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        for item in items:
            self._queue.put_nowait(item)


def test_backlog_is_skipped_to_the_newest_frame() -> None:
    track = FakeTrack(["f2", "f3", "f4", "f5"])
    frame, skipped = skip_to_newest(track, "f1")  # type: ignore[arg-type]
    assert (frame, skipped) == ("f5", 4)
    assert track._queue.empty()


def test_small_backlog_is_left_alone() -> None:
    track = FakeTrack(["f2"])
    assert skip_to_newest(track, "f1") == ("f1", 0)  # type: ignore[arg-type]
    assert track._queue.qsize() == 1


def test_end_of_stream_marker_survives_skipping() -> None:
    track = FakeTrack(["f2", "f3", "f4", None])
    frame, skipped = skip_to_newest(track, "f1")  # type: ignore[arg-type]
    assert (frame, skipped) == ("f4", 3)
    assert track._queue.get_nowait() is None  # recv() still sees the end of the stream
