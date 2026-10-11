import asyncio

import numpy as np

from pocketslam.config import Settings
from pocketslam.core.types import ImuSample, Intrinsics, LinkStats, SessionMetadata
from pocketslam.pipeline import create_session


async def test_processes_frames_and_writes_trajectory(settings: Settings) -> None:
    session = create_session(settings)
    session.start()
    for i in range(3):
        session.on_frame(np.zeros((48, 64, 3), dtype=np.uint8), float(i))
        await asyncio.sleep(0.05)
    await session.close()
    assert session.processed == 3
    trajectory = settings.output_dir / session.session_id / "trajectory.tum"
    assert len(trajectory.read_text().splitlines()) == 3


async def test_uses_fov_estimate_without_metadata(settings: Settings) -> None:
    session = create_session(settings)
    session.on_frame(np.zeros((48, 64, 3), dtype=np.uint8), 0.0)
    frame = await session.queue.get()
    assert frame is not None
    assert frame.intrinsics.source == "fov-estimate"
    await session.close()


async def test_scales_device_intrinsics_to_frame_size(settings: Settings) -> None:
    session = create_session(settings)
    k = Intrinsics(1000.0, 1000.0, 640.0, 360.0, 1280, 720, source="device")
    session.on_metadata(SessionMetadata(1280, 720, "phone", k))
    session.on_frame(np.zeros((360, 640, 3), dtype=np.uint8), 0.0)
    frame = await session.queue.get()
    assert frame is not None
    assert frame.intrinsics.fx == 500.0
    assert frame.intrinsics.source == "device"
    await session.close()


async def test_imu_is_counted_and_mapped_to_host_clock(settings: Settings) -> None:
    session = create_session(settings)
    session.on_imu([ImuSample("gyro", 100.0, 0, 0, 1), ImuSample("accel", 100.01, 0, 9.8, 0)])
    assert session.imu_received == 2
    assert session.stats()["imu_received"] == 2
    await session.close()


async def test_close_twice_is_safe(settings: Settings) -> None:
    session = create_session(settings)
    session.start()
    await asyncio.gather(session.close(), session.close())
    await session.close()


async def test_link_stats_and_receive_rate_in_status(settings: Settings) -> None:
    session = create_session(settings)
    session.on_stats(LinkStats(camera_fps=30.0, sent_fps=12.0, quality_limitation="bandwidth"))
    for i in range(5):
        session.on_frame(np.zeros((48, 64, 3), dtype=np.uint8), i / 30)
    stats = session.stats()
    assert stats["recv_fps"] == 5.0
    assert stats["link"] == {
        "camera_fps": 30.0,
        "sent_fps": 12.0,
        "width": None,
        "height": None,
        "quality_limitation": "bandwidth",
        "encoder": None,
        "codec": None,
        "encode_ms": None,
        "target_kbps": None,
        "available_kbps": None,
        "rtt_ms": None,
        "nack": None,
        "pli": None,
    }
    await session.close()
