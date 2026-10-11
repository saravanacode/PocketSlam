import json
from pathlib import Path

from mcap.reader import make_reader

from pocketslam.core.types import ImuSample, Intrinsics, SessionMetadata
from pocketslam.record.telemetry import (
    TOPIC_ACCEL,
    TOPIC_CALIBRATION,
    TOPIC_GYRO,
    TOPIC_IMAGE,
    TOPIC_META,
    Telemetry,
    to_ns,
)


def test_logs_every_topic_to_mcap(tmp_path: Path) -> None:
    telemetry = Telemetry()
    path = tmp_path / "rec" / "s.mcap"
    writer = telemetry.open_recording(path)
    k = Intrinsics.from_fov(64, 48, 60.0)
    telemetry.log_meta(SessionMetadata(64, 48, "phone"), 10.0)
    telemetry.log_calibration(k, 10.0)
    telemetry.log_image(b"\xff\xd8fake", 10.0)
    telemetry.log_imu(ImuSample("gyro", 1.0, 0.1, 0.2, 0.3), 10.5)
    telemetry.log_imu(ImuSample("accel", 1.0, 0.0, 9.8, 0.0), 10.5)
    writer.close()

    with path.open("rb") as f:
        messages = {ch.topic: (msg, ch) for _, ch, msg in make_reader(f).iter_messages()}
    assert set(messages) == {TOPIC_META, TOPIC_CALIBRATION, TOPIC_IMAGE, TOPIC_GYRO, TOPIC_ACCEL}
    gyro_msg, gyro_ch = messages[TOPIC_GYRO]
    assert gyro_ch.message_encoding == "json"
    assert gyro_msg.log_time == to_ns(10.5)
    assert json.loads(gyro_msg.data)["x"] == 0.1
    assert messages[TOPIC_IMAGE][1].message_encoding == "protobuf"


def test_live_server_starts_and_stops() -> None:
    telemetry = Telemetry()
    port = telemetry.start_server("127.0.0.1", 0)
    assert port > 0
    telemetry.stop_server()
    telemetry.stop_server()  # second stop is a no-op
