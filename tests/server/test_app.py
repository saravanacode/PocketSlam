import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from pocketslam.config import Settings
from pocketslam.server.app import SESSIONS_KEY, create_app, create_ssl_context


async def test_healthz(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        resp = await client.get("/healthz")
        assert resp.status == 200
        assert await resp.json() == {"status": "ok"}


async def test_serves_phone_client(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        resp = await client.get("/")
        assert resp.status == 200
        assert "getUserMedia" in await (await client.get("/static/app.js")).text()


async def test_offer_rejects_bad_body(settings: Settings) -> None:
    async with TestClient(TestServer(create_app(settings))) as client:
        assert (await client.post("/offer", data="nope")).status == 400
        assert (await client.post("/offer", json={"type": "answer", "sdp": "x"})).status == 400


async def test_phone_streams_frames_end_to_end(settings: Settings) -> None:
    app = create_app(settings)
    async with TestClient(TestServer(app)) as client:
        phone = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        phone.addTrack(VideoStreamTrack())
        channel = phone.createDataChannel("meta")
        channel.on(
            "open", lambda: channel.send(json.dumps({"type": "meta", "width": 640, "height": 480}))
        )
        await phone.setLocalDescription(await phone.createOffer())

        resp = await client.post(
            "/offer", json={"sdp": phone.localDescription.sdp, "type": phone.localDescription.type}
        )
        assert resp.status == 200
        answer = await resp.json()
        await phone.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )

        session = app[SESSIONS_KEY][answer["session_id"]]
        for _ in range(100):
            if session.processed >= 3 and session.metadata is not None:
                break
            await asyncio.sleep(0.05)
        assert session.processed >= 3
        assert session.metadata is not None
        assert session.metadata.width == 640
        await phone.close()


def test_no_ssl_context_without_cert(settings: Settings) -> None:
    assert create_ssl_context(settings) is None
