import asyncio
import json
import os
import socket
from pathlib import Path

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from pocketslam.config import Settings
from pocketslam.core.types import ImuSample
from pocketslam.server.dashboard import (
    LiveHub,
    create_dashboard_app,
    phone_url,
    start_on_first_free_port,
)


async def test_streams_status_frames_and_imu(settings: Settings) -> None:
    hub = LiveHub()
    app = create_dashboard_app(
        settings, hub, status=lambda: {"phone_url": "https://x:10001", "session": None}
    )
    async with TestClient(TestServer(app)) as client:
        assert "PocketSlam" in await (await client.get("/")).text()
        assert (await client.get("/static/app.js")).status == 200
        assert "WebGL" in await (await client.get("/static/scans.js")).text()
        ws = await client.ws_connect("/ws")

        status = json.loads((await ws.receive(timeout=2)).data)
        assert status["type"] == "status"
        assert status["phone_url"] == "https://x:10001"
        assert status["session"] is None
        assert status["loop_lag_ms"] >= 0

        # Publish from another thread, as the recording worker does.
        await asyncio.to_thread(hub.publish_imu, [(ImuSample("gyro", 0.0, 0.1, 0.2, 0.3), 5.0)])
        await asyncio.to_thread(hub.publish_frame, b"\xff\xd8jpeg", 5.0)
        got: dict[str, object] = {}
        while len(got) < 2:
            msg = await ws.receive(timeout=2)
            if msg.type == WSMsgType.BINARY:
                got["frame"] = msg.data
            elif json.loads(msg.data)["type"] == "imu":
                got["imu"] = json.loads(msg.data)["samples"]
        assert got == {"frame": b"\xff\xd8jpeg", "imu": [["g", 5.0, 0.1, 0.2, 0.3]]}
        await ws.close()


def test_publish_before_bind_is_noop() -> None:
    hub = LiveHub()
    hub.publish_frame(b"x", 0.0)
    hub.publish_imu([(ImuSample("accel", 0.0, 0, 0, 0), 0.0)])
    assert hub.client_count == 0


async def test_falls_back_to_next_free_port() -> None:
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    busy = blocker.getsockname()[1]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    runner = web.AppRunner(web.Application())
    await runner.setup()
    try:
        assert await start_on_first_free_port(runner, "127.0.0.1", [busy, free]) == free
        with pytest.raises(OSError, match="in use"):
            await start_on_first_free_port(runner, "127.0.0.1", [busy])
    finally:
        await runner.cleanup()
        blocker.close()


def test_phone_url_uses_lan_ip_for_wildcard(settings: Settings) -> None:
    s = settings.model_copy(update={"host": "192.168.1.5", "port": 10001})
    assert phone_url(s) == "http://192.168.1.5:10001"
    wildcard = settings.model_copy(update={"host": "0.0.0.0"})  # noqa: S104 - test input
    assert not phone_url(wildcard).startswith("http://0.0.0.0")


def _make_scan(root: Path, name: str, points: int = 2) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "points.ply").write_bytes(b"ply\nend_header\n" + b"\0" * 15 * points)
    (d / "trajectory.tum").write_text("1.0 0 0 0 0 0 0 1\n")
    (d / "summary.json").write_text(
        json.dumps(
            {"points": points, "scale": 1.4, "scale_source": "imu", "imu": {"observable": True}}
        )
    )


async def test_lists_and_serves_scans(settings: Settings, tmp_path: Path) -> None:
    root = tmp_path / "recon"
    _make_scan(root, "older")
    _make_scan(root, "newer", points=5)
    os.utime(root / "older" / "points.ply", (1, 1))
    (root / "no_summary").mkdir()
    (root / "no_summary" / "points.ply").write_bytes(b"ply")
    s = settings.model_copy(update={"reconstructions_dir": root})
    app = create_dashboard_app(s, LiveHub(), status=dict)
    async with TestClient(TestServer(app)) as client:
        scans = await (await client.get("/api/scans")).json()
        assert [x["name"] for x in scans] == ["newer", "older"]
        assert scans[0]["points"] == 5
        assert scans[0]["imu_observable"] is True
        ply = await client.get("/api/scans/newer/points.ply")
        assert ply.status == 200
        assert (await ply.read()).startswith(b"ply")
        assert (await client.get("/api/scans/newer/trajectory.tum")).status == 200


@pytest.mark.parametrize(
    "path",
    [
        "/api/scans/newer/depth.npz",
        "/api/scans/missing/points.ply",
        "/api/scans/..%2F..%2Fetc/points.ply",
        "/api/scans/.hidden/points.ply",
    ],
)
async def test_scan_files_are_restricted(settings: Settings, tmp_path: Path, path: str) -> None:
    root = tmp_path / "recon"
    _make_scan(root, "newer")
    s = settings.model_copy(update={"reconstructions_dir": root})
    async with TestClient(TestServer(create_dashboard_app(s, LiveHub(), status=dict))) as client:
        assert (await client.get(path)).status == 404


async def test_scans_empty_when_folder_missing(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"reconstructions_dir": tmp_path / "none"})
    async with TestClient(TestServer(create_dashboard_app(s, LiveHub(), status=dict))) as client:
        assert await (await client.get("/api/scans")).json() == []


async def test_record_endpoint(settings: Settings) -> None:
    calls: list[bool] = []

    class Button:
        async def set(self, on: bool) -> dict[str, object]:
            calls.append(on)
            return {"clip": {"path": "data/recordings/clip_x.mcap"} if on else None}

        async def calibrate(self, on: bool) -> dict[str, object]:
            return {"active": on, "views": 0}

        async def demo(self, on: bool, recalibrate: bool = False) -> dict[str, object]:
            return {"stage": "calibrating" if recalibrate else "measuring" if on else "idle"}

        def jobs(self) -> list[dict[str, object]]:
            return [{"name": "clip_x", "state": "running"}]

    app = create_dashboard_app(settings, LiveHub(), status=dict, record_button=Button())
    async with TestClient(TestServer(app)) as client:
        cal = await client.post("/api/calibrate", json={"on": True})
        assert await cal.json() == {"active": True, "views": 0}
        assert (await client.post("/api/calibrate", json={})).status == 400
        started = await client.post("/api/demo", json={"on": True})
        assert await started.json() == {"stage": "measuring"}
        again = await client.post("/api/demo", json={"on": True, "recalibrate": True})
        assert await again.json() == {"stage": "calibrating"}
        assert (
            await client.post("/api/demo", json={"on": True, "recalibrate": "yes"})
        ).status == 400
        resp = await client.post("/api/record", json={"on": True})
        assert resp.status == 200
        assert (await resp.json())["clip"]["path"].endswith("clip_x.mcap")
        assert (await client.post("/api/record", json={"on": "yes"})).status == 400
        assert (await client.post("/api/record", data="x")).status == 400
        assert (await (await client.get("/api/jobs")).json())[0]["state"] == "running"
    assert calls == [True]


async def test_record_endpoint_unavailable(settings: Settings) -> None:
    async with TestClient(
        TestServer(create_dashboard_app(settings, LiveHub(), status=dict))
    ) as client:
        assert (await client.post("/api/record", json={"on": True})).status == 409
        assert (await client.post("/api/calibrate", json={"on": True})).status == 409
        assert (await client.post("/api/demo", json={"on": True})).status == 409
        assert await (await client.get("/api/jobs")).json() == []


async def test_hub_publish_json_from_thread(settings: Settings) -> None:
    hub = LiveHub()
    app = create_dashboard_app(settings, hub, status=lambda: {"phone_url": "x", "session": None})
    async with TestClient(TestServer(app)) as client:
        ws = await client.ws_connect("/ws")
        await ws.receive(timeout=2)  # first status
        await asyncio.to_thread(hub.publish_json, {"type": "calibration", "views": 3})
        for _ in range(5):
            msg = json.loads((await ws.receive(timeout=2)).data)
            if msg["type"] == "calibration":
                break
        assert msg == {"type": "calibration", "views": 3}
        await ws.close()


async def test_scan_with_nan_summary_does_not_break_the_list(
    settings: Settings, tmp_path: Path
) -> None:
    """Regression: one scan with NaN in summary.json hid every scan in the browser."""
    root = tmp_path / "recon"
    _make_scan(root, "good")
    _make_scan(root, "degenerate")
    (root / "degenerate" / "summary.json").write_text(
        '{"points": 1, "scale": 0.8, "scale_source": "metric-depth", "path_length_m": NaN,'
        ' "imu": {"observable": false, "accel_std": NaN}}'
    )
    s = settings.model_copy(update={"reconstructions_dir": root})

    def strict(text: str) -> object:
        def reject(token: str) -> None:
            raise AssertionError(f"invalid JSON token {token}")

        return json.loads(text, parse_constant=reject)

    async with TestClient(TestServer(create_dashboard_app(s, LiveHub(), status=dict))) as client:
        listing = strict(await (await client.get("/api/scans")).text())
        assert isinstance(listing, list)
        by_name = {x["name"]: x for x in listing}
        assert set(by_name) == {"good", "degenerate"}
        assert by_name["degenerate"]["path_length_m"] is None
        summary = strict(await (await client.get("/api/scans/degenerate/summary.json")).text())
        assert summary["imu"]["accel_std"] is None  # type: ignore[index]


async def test_status_stream_never_sends_nan(settings: Settings) -> None:
    hub = LiveHub()
    app = create_dashboard_app(
        settings, hub, status=lambda: {"phone_url": "x", "bad": float("nan")}
    )
    async with TestClient(TestServer(app)) as client:
        ws = await client.ws_connect("/ws")
        text = (await ws.receive(timeout=2)).data
        assert "NaN" not in text
        assert json.loads(text)["bad"] is None
        await ws.close()
