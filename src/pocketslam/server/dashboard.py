"""Laptop dashboard: live camera, IMU plots and recording status over WebSocket.

The dashboard is a second aiohttp app bound to localhost. ``LiveHub`` receives
frames and IMU from the recording thread and fans them out to every open
dashboard tab. Each tab gets the newest frame only, so a slow tab never
backs up the recorder.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import re
import socket
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from pocketslam.config import Settings
from pocketslam.core.jsonsafe import dumps
from pocketslam.core.types import ImuSample

logger = logging.getLogger(__name__)

STATUS_PERIOD_S = 0.5
MAX_PENDING_TEXT = 256


class _Client:
    """One dashboard tab: newest frame plus a bounded queue of JSON messages."""

    def __init__(self, ws: web.WebSocketResponse) -> None:
        self.ws = ws
        self.frame: bytes | None = None
        self.texts: deque[str] = deque(maxlen=MAX_PENDING_TEXT)
        self.ready = asyncio.Event()

    async def run(self) -> None:
        try:
            while not self.ws.closed:
                await self.ready.wait()
                self.ready.clear()
                while self.texts:
                    await self.ws.send_str(self.texts.popleft())
                if self.frame is not None:
                    frame, self.frame = self.frame, None
                    await self.ws.send_bytes(frame)
        except ConnectionResetError:
            return  # tab closed mid-send


class LiveHub:
    """Fans live data out to dashboard clients.

    ``publish_frame`` and ``publish_imu`` are safe to call from any thread.
    Before ``bind`` is called they do nothing.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[_Client] = set()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the event loop that owns the WebSocket connections."""
        self._loop = loop

    @property
    def client_count(self) -> int:
        """Number of connected dashboard tabs."""
        return len(self._clients)

    def publish_frame(self, jpeg: bytes, t_s: float) -> None:
        """Send a JPEG frame to every client (newest frame wins)."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._set_frame, jpeg)

    def publish_imu(self, samples: list[tuple[ImuSample, float]]) -> None:
        """Send IMU samples to every client as one JSON message."""
        if self._loop is None or not samples:
            return
        payload = json.dumps(
            {
                "type": "imu",
                "samples": [
                    [s.kind[0], round(t, 4), round(s.x, 4), round(s.y, 4), round(s.z, 4)]
                    for s, t in samples
                ],
            }
        )
        self._loop.call_soon_threadsafe(self.publish_text, payload)

    def publish_json(self, data: dict[str, Any]) -> None:
        """Send a JSON message to every client. Safe to call from any thread."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.publish_text, dumps(data))

    def publish_text(self, payload: str) -> None:
        """Queue a JSON message for every client. Event-loop thread only."""
        for client in self._clients:
            client.texts.append(payload)
            client.ready.set()

    def _set_frame(self, jpeg: bytes) -> None:
        for client in self._clients:
            client.frame = jpeg
            client.ready.set()

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        """Serve one dashboard WebSocket until it closes."""
        # No compression: JPEG frames do not shrink, and deflate runs in a
        # background task that errors if the tab closes mid-send.
        ws = web.WebSocketResponse(heartbeat=10.0, compress=False)
        await ws.prepare(request)
        client = _Client(ws)
        self._clients.add(client)
        sender = asyncio.create_task(client.run())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(client)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, ConnectionResetError):
                await sender
        return ws

    async def close_all(self) -> None:
        """Close every client connection."""
        for client in list(self._clients):
            await client.ws.close()


SETTINGS_KEY = web.AppKey("dashboard_settings", Settings)
HUB_KEY = web.AppKey("live_hub", LiveHub)
STATUS_KEY: web.AppKey[Callable[[], dict[str, Any]]] = web.AppKey("status_provider")
RECORD_KEY: web.AppKey[Any] = web.AppKey("record_button")
STATUS_TASK_KEY = web.AppKey("status_task", asyncio.Task[None])


def lan_ip() -> str:
    """Best guess at this machine's LAN address. No packets are sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Connecting a UDP socket only picks a route; nothing goes on the wire.
            s.connect(("10.255.255.255", 1))
            return str(s.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def phone_url(settings: Settings) -> str:
    """URL to open on the phone."""
    scheme = "https" if settings.tls_enabled else "http"
    host = lan_ip() if settings.host in {"0.0.0.0", "::"} else settings.host  # noqa: S104 - comparing
    return f"{scheme}://{host}:{settings.port}"


async def index(request: web.Request) -> web.FileResponse:
    """Serve the dashboard page."""
    return web.FileResponse(request.app[SETTINGS_KEY].web_dir / "dashboard" / "index.html")


async def _status_loop(app: web.Application) -> None:
    hub = app[HUB_KEY]
    provider = app[STATUS_KEY]
    loop = asyncio.get_running_loop()
    lag_ms = 0.0
    while True:
        if hub.client_count:
            status = {"type": "status", "loop_lag_ms": round(lag_ms, 1), **provider()}
            hub.publish_text(dumps(status))
        start = loop.time()
        await asyncio.sleep(STATUS_PERIOD_S)
        # How late the sleep woke up: time the event loop spent blocked.
        lag_ms = max(0.0, (loop.time() - start - STATUS_PERIOD_S) * 1000)


async def _on_startup(app: web.Application) -> None:
    app[HUB_KEY].bind(asyncio.get_running_loop())
    app[STATUS_TASK_KEY] = asyncio.create_task(_status_loop(app))


async def _on_shutdown(app: web.Application) -> None:
    app[STATUS_TASK_KEY].cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await app[STATUS_TASK_KEY]
    await app[HUB_KEY].close_all()


SCAN_FILES = {"points.ply", "trajectory.tum", "summary.json"}
_SCAN_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def list_scans(root: Path) -> list[dict[str, Any]]:
    """Reconstructions under ``root`` that have a point cloud, newest first."""
    if not root.is_dir():
        return []
    scans = []
    for d in root.iterdir():
        ply, summary_path = d / "points.ply", d / "summary.json"
        if not (_SCAN_NAME.match(d.name) and ply.is_file() and summary_path.is_file()):
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("unreadable scan summary", extra={"scan": d.name})
            continue
        mtime = ply.stat().st_mtime
        scans.append(
            {
                "name": d.name,
                "created": datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
                "points": summary.get("points"),
                "scale": summary.get("scale"),
                "scale_source": summary.get("scale_source"),
                "clip_seconds": summary.get("clip_seconds"),
                "keyframes": summary.get("keyframes"),
                "path_length_m": summary.get("path_length_m"),
                "imu_observable": (summary.get("imu") or {}).get("observable"),
                "size_bytes": ply.stat().st_size,
                "_mtime": mtime,
            }
        )
    scans.sort(key=lambda s: s.pop("_mtime"), reverse=True)
    return scans


async def scans_index(request: web.Request) -> web.Response:
    """JSON list of reconstructions for the 3D tab."""
    scans = list_scans(request.app[SETTINGS_KEY].reconstructions_dir)
    return web.json_response(scans, dumps=dumps)


async def scan_file(request: web.Request) -> web.StreamResponse:
    """Serve one file of one reconstruction; only known file names, no path tricks."""
    name, file = request.match_info["name"], request.match_info["file"]
    if not _SCAN_NAME.match(name) or file not in SCAN_FILES:
        raise web.HTTPNotFound
    root = request.app[SETTINGS_KEY].reconstructions_dir.resolve()
    path = (root / name / file).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise web.HTTPNotFound
    if file == "summary.json":
        # Older summaries can hold NaN, which browsers cannot parse; clean on the way out.
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise web.HTTPNotFound from exc
        return web.json_response(summary, dumps=dumps, headers={"Cache-Control": "no-cache"})
    return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


async def record(request: web.Request) -> web.Response:
    """Start or stop a clip: body ``{"on": true}`` or ``{"on": false}``."""
    if RECORD_KEY not in request.app:
        raise web.HTTPConflict(text="Recording is not available on this server.")
    try:
        body: Any = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text="body must be JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("on"), bool):
        raise web.HTTPBadRequest(text="expected {'on': true|false}")
    return web.json_response(await request.app[RECORD_KEY].set(body["on"]), dumps=_dumps)


async def calibrate(request: web.Request) -> web.Response:
    """Start or stop live camera calibration: body ``{"on": true}`` or ``{"on": false}``."""
    if RECORD_KEY not in request.app:
        raise web.HTTPConflict(text="Calibration is not available on this server.")
    try:
        body: Any = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text="body must be JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("on"), bool):
        raise web.HTTPBadRequest(text="expected {'on': true|false}")
    return web.json_response(await request.app[RECORD_KEY].calibrate(body["on"]), dumps=_dumps)


async def demo(request: web.Request) -> web.Response:
    """Start or stop the distance demo.

    Body: ``{"on": true|false, "recalibrate": true|false}``; ``recalibrate`` is optional.
    """
    if RECORD_KEY not in request.app:
        raise web.HTTPConflict(text="The distance demo is not available on this server.")
    try:
        body: Any = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text="body must be JSON") from exc
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("on"), bool)
        or not isinstance(body.get("recalibrate", False), bool)
    ):
        raise web.HTTPBadRequest(text="expected {'on': true|false, 'recalibrate': true|false}")
    result = await request.app[RECORD_KEY].demo(body["on"], body.get("recalibrate", False))
    return web.json_response(result, dumps=_dumps)


async def jobs(request: web.Request) -> web.Response:
    """Reconstruction jobs, newest first."""
    listed = request.app[RECORD_KEY].jobs() if RECORD_KEY in request.app else []
    return web.json_response(listed)


def _dumps(obj: Any) -> str:
    return dumps(obj)


def create_dashboard_app(
    settings: Settings,
    hub: LiveHub,
    status: Callable[[], dict[str, Any]],
    record_button: Any | None = None,
) -> web.Application:
    """Build the dashboard application.

    Args:
        settings: Runtime settings.
        hub: Live data fan-out shared with the phone-facing app.
        status: Returns the current status snapshot, sent to tabs twice a second.
        record_button: Starts and stops clips and lists reconstruction jobs
            (``server.app.RecordButton``); None hides recording.

    Returns:
        The configured application.
    """
    app = web.Application()
    app[SETTINGS_KEY] = settings
    app[HUB_KEY] = hub
    app[STATUS_KEY] = status
    if record_button is not None:
        app[RECORD_KEY] = record_button
    app.router.add_get("/", index)
    app.router.add_get("/ws", hub.handle)
    app.router.add_post("/api/record", record)
    app.router.add_post("/api/calibrate", calibrate)
    app.router.add_post("/api/demo", demo)
    app.router.add_get("/api/jobs", jobs)
    app.router.add_get("/api/scans", scans_index)
    app.router.add_get("/api/scans/{name}/{file}", scan_file)
    app.router.add_static("/static/", settings.web_dir / "dashboard", show_index=False)
    app.on_startup.append(_on_startup)
    app.on_shutdown.append(_on_shutdown)
    return app


async def start_on_first_free_port(runner: web.AppRunner, host: str, ports: list[int]) -> int:
    """Start a TCP site on the first port in ``ports`` that is free.

    Args:
        runner: A set-up runner.
        host: Interface to bind.
        ports: Ports to try, in order.

    Returns:
        The port the site listens on.

    Raises:
        OSError: If every port is in use, or binding fails for another reason.
    """
    for port in ports:
        site = web.TCPSite(runner, host, port)
        try:
            await site.start()
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            logger.warning("dashboard port in use, trying next", extra={"port": port})
            continue
        return port
    msg = f"all dashboard ports are in use: {ports}"
    raise OSError(errno.EADDRINUSE, msg)
