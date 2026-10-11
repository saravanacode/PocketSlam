import asyncio
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from PIL import Image

from pocketslam.calib.demo import DistanceDemo
from pocketslam.calib.distance import LiveDistance
from pocketslam.calib.live import LiveCalibration
from pocketslam.calib.marker import make_detector
from pocketslam.calib.store import ScaleCalibration
from pocketslam.config import Settings
from pocketslam.record.telemetry import TOPIC_ACCEL, TOPIC_GYRO, TOPIC_IMAGE, TOPIC_META, Telemetry
from pocketslam.server.app import (
    SESSIONS_KEY,
    RecordButton,
    create_app,
    create_ssl_context,
    session_status,
)

from ..record.conftest import read_topics


async def test_healthz(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        assert await resp.json() == {"status": "ok"}


async def test_serves_phone_client(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        resp = await client.get("/")
        assert resp.status == 200
        assert "PocketSlam" in await resp.text()
        assert "getUserMedia" in await (await client.get("/static/app.js")).text()
        assert (await client.get("/static/style.css")).status == 200


async def test_offer_rejects_bad_body(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        assert (await client.post("/offer", data="nope")).status == 400
        assert (await client.post("/offer", json={"type": "answer", "sdp": "x"})).status == 400


async def _connect_phone(client: TestClient, send_imu: bool) -> tuple[RTCPeerConnection, str]:
    phone = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    phone.addTrack(VideoStreamTrack())
    channel = phone.createDataChannel("data")

    def _on_open() -> None:
        channel.send(json.dumps({"type": "meta", "width": 640, "height": 480}))
        if send_imu:
            channel.send(
                json.dumps(
                    {
                        "type": "imu",
                        "samples": [["g", 1000.0, 0.1, 0.2, 0.3], ["a", 1000.0, 0.0, 9.8, 0.0]],
                    }
                )
            )

    channel.on("open", _on_open)
    await phone.setLocalDescription(await phone.createOffer())
    resp = await client.post(
        "/offer", json={"sdp": phone.localDescription.sdp, "type": phone.localDescription.type}
    )
    assert resp.status == 200
    answer = await resp.json()
    await phone.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    return phone, answer["session_id"]


async def _wait_for(predicate: object, timeout_s: float = 5.0) -> None:
    assert callable(predicate)
    for _ in range(int(timeout_s / 0.05)):
        if predicate():
            return
        await asyncio.sleep(0.05)


async def test_phone_streams_frames_end_to_end(settings: Settings) -> None:
    app = create_app(settings)
    async with TestClient(TestServer(app)) as client:
        phone, session_id = await _connect_phone(client, send_imu=False)
        session = app[SESSIONS_KEY][session_id]
        await _wait_for(lambda: session.processed >= 3 and session.metadata is not None)
        assert session.processed >= 3
        assert session.metadata is not None
        assert session.metadata.width == 640
        assert session_status(app)["session"]["frames_received"] >= 3  # type: ignore[index]
        await phone.close()


async def test_records_camera_and_imu_to_mcap(settings: Settings, tmp_path: Path) -> None:
    settings = settings.model_copy(update={"recordings_dir": tmp_path / "rec", "record": True})
    app = create_app(settings, Telemetry())
    async with TestClient(TestServer(app)) as client:
        phone, session_id = await _connect_phone(client, send_imu=True)
        session = app[SESSIONS_KEY][session_id]
        await _wait_for(lambda: session.imu_received >= 2 and session.processed >= 3)
        assert session.recorder is not None
        path = session.recorder.path
        await phone.close()
    assert path is not None
    topics = read_topics(path)
    assert topics[TOPIC_IMAGE] >= 3
    assert topics[TOPIC_GYRO] == topics[TOPIC_ACCEL] == topics[TOPIC_META] == 1


async def test_new_phone_replaces_old_session(settings: Settings) -> None:
    app = create_app(settings)
    async with TestClient(TestServer(app)) as client:
        first, first_id = await _connect_phone(client, send_imu=False)
        second, second_id = await _connect_phone(client, send_imu=False)
        assert list(app[SESSIONS_KEY]) == [second_id]
        assert first_id != second_id
        await first.close()
        await second.close()


def test_no_ssl_context_without_cert(settings: Settings) -> None:
    assert create_ssl_context(settings) is None


async def test_record_button_over_webrtc_saves_and_queues_clip(
    settings: Settings, tmp_path: Path
) -> None:
    s = settings.model_copy(update={"recordings_dir": tmp_path / "rec"})
    queued: list[Any] = []

    class FakeJobs:
        def start(self) -> None: ...
        async def stop(self) -> None: ...
        def snapshot(self) -> list[dict[str, Any]]:
            return []

        def submit(self, clip: Any) -> None:
            queued.append(clip)

    app = create_app(s, Telemetry(), jobs=FakeJobs())  # type: ignore[arg-type]
    async with TestClient(TestServer(app)) as client:
        phone, session_id = await _connect_phone(client, send_imu=False)
        session = app[SESSIONS_KEY][session_id]
        button = RecordButton(app)
        await _wait_for(lambda: session.processed >= 3)
        await button.set(True)
        frames_at_start = session.processed
        await _wait_for(lambda: session.processed >= frames_at_start + 12)
        state = await button.set(False)
        assert state["clip"] is None
        await phone.close()
    assert len(queued) == 1
    assert queued[0].frames >= 10
    assert read_topics(queued[0].path)[TOPIC_IMAGE] == queued[0].frames


async def test_record_button_without_phone_conflicts(settings: Settings) -> None:
    app = create_app(settings, Telemetry())
    with pytest.raises(web.HTTPConflict):
        await RecordButton(app).set(True)


async def test_live_calibration_receives_stream_frames(settings: Settings, tmp_path: Path) -> None:
    seen: list[tuple[int, ...]] = []

    def detector(image: Any) -> None:
        seen.append(image.shape)

    calibration = LiveCalibration(
        ScaleCalibration(tmp_path / "scale.json"), 0.16, 18, "DICT_4X4_50", detector=detector
    )
    app = create_app(settings, Telemetry(), calibration=calibration)
    async with TestClient(TestServer(app)) as client:
        button = RecordButton(app)
        with pytest.raises(web.HTTPConflict) as no_phone:
            await button.calibrate(True)
        assert "No phone" in (no_phone.value.text or "")
        phone, session_id = await _connect_phone(client, send_imu=False)
        session = app[SESSIONS_KEY][session_id]
        await _wait_for(lambda: session.processed >= 2)
        snap = await button.calibrate(True)
        assert snap["active"]
        await _wait_for(lambda: len(seen) >= 2)
        assert seen[0] == (480, 640, 3)
        status = session_status(app)
        assert status["calibration"]["hint"] == "Point the phone at the marker."  # type: ignore[index]
        assert status["focal_ratio"] is None
        assert not (await button.calibrate(False))["active"]
        await phone.close()


async def test_calibration_without_marker_size_explains(settings: Settings, tmp_path: Path) -> None:
    calibration = LiveCalibration(ScaleCalibration(tmp_path / "s.json"), None, None, "DICT_4X4_50")
    app = create_app(settings, Telemetry(), calibration=calibration)
    async with TestClient(TestServer(app)) as client:
        phone, _ = await _connect_phone(client, send_imu=False)
        with pytest.raises(web.HTTPConflict) as no_size:
            await RecordButton(app).calibrate(True)
        assert "Marker size is not set" in (no_size.value.text or "")
        await phone.close()


async def test_marker_page_for_second_screen(settings: Settings, tmp_path: Path) -> None:
    calibration = LiveCalibration(ScaleCalibration(tmp_path / "s.json"), 0.16, 18, "DICT_4X4_50")
    app = create_app(settings, Telemetry(), calibration=calibration)
    async with TestClient(TestServer(app)) as client:
        page = await client.get("/marker")
        assert page.status == 200
        assert "background: #fff" in await page.text()
        assert await (await client.get("/marker.json")).json() == {
            "dictionary": "DICT_4X4_50",
            "id": 18,
            "size_m": 0.16,
        }


async def test_marker_png_has_white_border_and_is_detectable(
    settings: Settings, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    calibration = LiveCalibration(ScaleCalibration(tmp_path / "s.json"), 0.16, 18, "DICT_4X4_50")
    app = create_app(settings, Telemetry(), calibration=calibration)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/marker.png")
        assert resp.content_type == "image/png"
        with Image.open(io.BytesIO(await resp.read())) as img:
            pixels = np.asarray(img.convert("L"))
    assert (pixels[:200] == 255).all()
    found = make_detector("DICT_4X4_50", 18)(np.stack([pixels] * 3, axis=-1))
    assert found is not None


async def test_no_marker_routes_without_calibration(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        assert (await client.get("/marker")).status == 404


async def test_distance_demo_measures_from_stream(settings: Settings, tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    # A marker 0.16 m wide that looks 87 px wide at f = 870 px is 1.6 m away.
    corners = np.array([[277.0, 197.0], [364.0, 197.0], [364.0, 284.0], [277.0, 284.0]])
    store = ScaleCalibration(tmp_path / "scale.json")
    detector = lambda _image: (18, corners)  # noqa: E731 - tiny test double
    calibration = LiveCalibration(store, 0.16, 18, "DICT_4X4_50", detector=detector)
    demo = DistanceDemo(calibration, LiveDistance(0.16, 18, "DICT_4X4_50", detector=detector))
    app = create_app(settings, Telemetry(), calibration=calibration, demo=demo)
    async with TestClient(TestServer(app)) as client:
        button = RecordButton(app)
        with pytest.raises(web.HTTPConflict) as no_phone:
            await button.demo(True)
        assert "No phone" in (no_phone.value.text or "")

        phone, session_id = await _connect_phone(client, send_imu=False)
        session = app[SESSIONS_KEY][session_id]
        await _wait_for(lambda: session.metadata is not None and session.processed >= 2)
        # The test stream is 640x480; f = 870 px is 870/640 of the longer side.
        store.add_focal("unknown", 870 / 640, 0.017, "live")
        snap = await button.demo(True)
        assert snap["stage"] == "measuring"
        await _wait_for(lambda: demo.snapshot()["distance"]["distance_m"] is not None)
        status = session_status(app)["demo"]
        assert status["stage"] == "measuring"  # type: ignore[index]
        assert status["distance"]["distance_m"] == pytest.approx(1.6, rel=0.02)  # type: ignore[index]
        assert (await button.demo(False))["stage"] == "idle"
        await phone.close()


async def test_demo_calibrates_first_for_unknown_phone(settings: Settings, tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    store = ScaleCalibration(tmp_path / "scale.json")
    calibration = LiveCalibration(store, 0.16, 18, "DICT_4X4_50", detector=lambda _i: None)
    demo = DistanceDemo(
        calibration, LiveDistance(0.16, 18, "DICT_4X4_50", detector=lambda _i: None)
    )
    app = create_app(settings, Telemetry(), calibration=calibration, demo=demo)
    async with TestClient(TestServer(app)) as client:
        phone, _ = await _connect_phone(client, send_imu=False)
        assert (await RecordButton(app).demo(True))["stage"] == "calibrating"
        assert (await RecordButton(app).demo(False))["stage"] == "idle"
        await phone.close()
