"""Phone-facing HTTPS server: serves the phone client and handles WebRTC signaling."""

from __future__ import annotations

import asyncio
import io
import logging
import ssl
import time
from collections.abc import Callable
from typing import Any

from aiohttp import web

from pocketslam.calib.demo import DistanceDemo
from pocketslam.calib.live import LiveCalibration
from pocketslam.calib.store import device_label
from pocketslam.config import Settings
from pocketslam.pipeline import Session, start_session
from pocketslam.record.telemetry import Telemetry
from pocketslam.record.worker import LiveSink
from pocketslam.server.recording import ReconJobs, RecordingControl

logger = logging.getLogger(__name__)

SETTINGS_KEY = web.AppKey("settings", Settings)
SESSIONS_KEY = web.AppKey("sessions", dict[str, Session])
CLOSING_KEY = web.AppKey("closing", set[asyncio.Task[None]])
TELEMETRY_KEY = web.AppKey("telemetry", Telemetry)
LIVE_KEY: web.AppKey[LiveSink] = web.AppKey("live")
CONTROL_KEY = web.AppKey("record_control", RecordingControl)
JOBS_KEY = web.AppKey("recon_jobs", ReconJobs)
CALIBRATION_KEY = web.AppKey("live_calibration", LiveCalibration)
DEMO_KEY = web.AppKey("distance_demo", DistanceDemo)
LOOP_KEY = web.AppKey("event_loop", asyncio.AbstractEventLoop)
PHONE_DISTANCE_PERIOD_S = 0.2
MAX_OFFER_BYTES = 64 * 1024


async def healthz(_request: web.Request) -> web.Response:
    """Liveness probe used by Docker HEALTHCHECK."""
    return web.json_response({"status": "ok"})


async def index(request: web.Request) -> web.FileResponse:
    """Serve the phone client page."""
    return web.FileResponse(request.app[SETTINGS_KEY].web_dir / "phone" / "index.html")


async def marker_page(request: web.Request) -> web.FileResponse:
    """Page that shows the calibration marker on a white background, for a second screen."""
    return web.FileResponse(request.app[SETTINGS_KEY].web_dir / "phone" / "marker.html")


async def marker_info(request: web.Request) -> web.Response:
    """The configured marker (dictionary, id, size) for the marker page."""
    return web.json_response(request.app[CALIBRATION_KEY].marker)


async def marker_png(request: web.Request) -> web.Response:
    """The marker as a PNG with the white border detection needs."""
    marker = request.app[CALIBRATION_KEY].marker
    try:
        from PIL import Image  # noqa: PLC0415

        from pocketslam.calib.marker import render_marker  # noqa: PLC0415

        pixels = render_marker(marker["dictionary"], marker["id"] or 0, side_px=1000)
    except ImportError as exc:
        raise web.HTTPServiceUnavailable(
            text="Marker rendering needs OpenCV. Run: uv sync --extra recon"
        ) from exc
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return web.Response(body=buf.getvalue(), content_type="image/png")


async def offer(request: web.Request) -> web.Response:
    """Accept an SDP offer from a phone and return the SDP answer.

    Body: ``{"sdp": "...", "type": "offer"}``.
    """
    if request.content_length is not None and request.content_length > MAX_OFFER_BYTES:
        raise web.HTTPRequestEntityTooLarge(
            max_size=MAX_OFFER_BYTES, actual_size=request.content_length
        )
    try:
        body: Any = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text="body must be JSON") from exc
    if (
        not isinstance(body, dict)
        or body.get("type") != "offer"
        or not isinstance(body.get("sdp"), str)
    ):
        raise web.HTTPBadRequest(text="expected {'sdp': str, 'type': 'offer'}")

    sessions = request.app[SESSIONS_KEY]
    # One phone at a time: recordings and live topics are not per-device yet.
    # A new offer (for example a phone reconnecting) replaces the old session.
    if sessions:
        old = list(sessions.values())
        sessions.clear()
        logger.info("replacing previous session", extra={"count": len(old)})
        await asyncio.gather(*(s.close() for s in old), return_exceptions=True)

    closing = request.app[CLOSING_KEY]

    def _forget(session: Session) -> None:
        sessions.pop(session.session_id, None)
        # Keep a reference so shutdown can wait for the MCAP file to finish.
        task = asyncio.get_running_loop().create_task(session.close())
        closing.add(task)
        task.add_done_callback(closing.discard)

    session, peer = await start_session(
        request.app[SETTINGS_KEY],
        body["sdp"],
        body["type"],
        on_closed=_forget,
        telemetry=request.app.get(TELEMETRY_KEY),
        live=request.app.get(LIVE_KEY),
    )
    sessions[session.session_id] = session
    if CONTROL_KEY in request.app:
        request.app[CONTROL_KEY].attach(session)
    if DEMO_KEY in request.app:
        session.frame_tap = request.app[DEMO_KEY].submit  # feeds calibration and distance
    elif CALIBRATION_KEY in request.app:
        session.frame_tap = request.app[CALIBRATION_KEY].submit
    recording = session.recorder.path if session.recorder else None
    logger.info(
        "session started",
        extra={
            "session_id": session.session_id,
            "recording": str(recording) if recording else None,
        },
    )
    return web.json_response(
        {"sdp": peer.answer.sdp, "type": peer.answer.type, "session_id": session.session_id}
    )


async def _close_sessions(app: web.Application) -> None:
    sessions = list(app[SESSIONS_KEY].values())
    app[SESSIONS_KEY].clear()
    await asyncio.gather(*(s.close() for s in sessions), *app[CLOSING_KEY], return_exceptions=True)


async def _start_live_view(app: web.Application) -> None:
    settings = app[SETTINGS_KEY]
    app[TELEMETRY_KEY].start_server(settings.live_view_host, settings.live_view_port)


async def _stop_live_view(app: web.Application) -> None:
    app[TELEMETRY_KEY].stop_server()


async def _stop_demo(app: web.Application) -> None:
    await asyncio.to_thread(app[DEMO_KEY].stop)


async def _stop_calibration(app: web.Application) -> None:
    await asyncio.to_thread(app[CALIBRATION_KEY].stop)


def phone_calibration_notifier(app: web.Application) -> Callable[[dict[str, Any]], None]:
    """Build a callback that forwards calibration hints to the phone.

    The callback may run on any thread. It sends a message only when the hint
    or state changes, so the data channel is not flooded.
    """
    last: list[tuple[object, ...]] = [()]

    def send(snapshot: dict[str, Any]) -> None:
        key = (snapshot["hint"], snapshot["active"], snapshot["saved"], snapshot["views"])
        sessions = list(app[SESSIONS_KEY].values())
        if key == last[0] or not sessions:
            return
        last[0] = key
        sessions[0].notify_phone(
            {
                "type": "calibration",
                "active": snapshot["active"],
                "saved": snapshot["saved"],
                "hint": snapshot["hint"],
                "views": snapshot["views"],
            }
        )

    def notify(snapshot: dict[str, Any]) -> None:
        loop = app.get(LOOP_KEY)
        if loop is not None:
            loop.call_soon_threadsafe(send, snapshot)

    return notify


def phone_distance_notifier(app: web.Application) -> Callable[[dict[str, Any]], None]:
    """Build a callback that forwards the live distance to the phone, about 5 times a second."""
    last_sent = [0.0]

    def send(snapshot: dict[str, Any]) -> None:
        sessions = list(app[SESSIONS_KEY].values())
        if sessions:
            sessions[0].notify_phone(
                {
                    "type": "distance",
                    "active": snapshot["active"],
                    "visible": snapshot["marker_visible"],
                    "distance_m": snapshot["distance_m"],
                    "error_m": snapshot["error_m"],
                }
            )

    def notify(snapshot: dict[str, Any]) -> None:
        loop = app.get(LOOP_KEY)
        now = time.monotonic()
        # Always pass the final "stopped" update; throttle the rest.
        if loop is None or (snapshot["active"] and now - last_sent[0] < PHONE_DISTANCE_PERIOD_S):
            return
        last_sent[0] = now
        loop.call_soon_threadsafe(send, snapshot)

    return notify


async def _remember_loop(app: web.Application) -> None:
    app[LOOP_KEY] = asyncio.get_running_loop()


def _session_device(session: Session) -> str:
    meta = session.metadata
    return device_label({"model": meta.model, "device": meta.device_label} if meta else None)


async def _start_jobs(app: web.Application) -> None:
    app[JOBS_KEY].start()


async def _stop_jobs(app: web.Application) -> None:
    await app[JOBS_KEY].stop()


def create_app(
    settings: Settings,
    telemetry: Telemetry | None = None,
    *,
    live: LiveSink | None = None,
    jobs: ReconJobs | None = None,
    calibration: LiveCalibration | None = None,
    demo: DistanceDemo | None = None,
) -> web.Application:
    """Build the aiohttp application.

    Args:
        settings: Runtime settings.
        telemetry: Foxglove channels for live view and MCAP recording. None
            disables both. When ``settings.live_view`` is set, the Foxglove
            WebSocket server starts and stops with the app.
        live: Receives encoded frames and IMU for the laptop dashboard.
        jobs: Reconstructs saved clips. None disables automatic reconstruction.
        calibration: Live focal-length calibration fed by the phone's frames.
        demo: Distance demo (calibrate, then live distance to the marker). It
            wraps ``calibration`` and feeds it frames.

    Returns:
        The configured application.
    """
    app = web.Application(client_max_size=MAX_OFFER_BYTES)
    app[SETTINGS_KEY] = settings
    app[SESSIONS_KEY] = {}
    app[CLOSING_KEY] = set()
    if live is not None:
        app[LIVE_KEY] = live
    if demo is not None:
        app[DEMO_KEY] = demo
        app.on_cleanup.append(_stop_demo)
    if calibration is not None:
        app[CALIBRATION_KEY] = calibration
        app.router.add_get("/marker", marker_page)
        app.router.add_get("/marker.json", marker_info)
        app.router.add_get("/marker.png", marker_png)
        app.on_startup.append(_remember_loop)
        app.on_cleanup.append(_stop_calibration)
    if jobs is not None:
        app[JOBS_KEY] = jobs
        app.on_startup.append(_start_jobs)
        app.on_cleanup.append(_stop_jobs)
    if telemetry is not None:
        app[TELEMETRY_KEY] = telemetry
        app[CONTROL_KEY] = RecordingControl(settings, jobs)
        if settings.live_view:
            app.on_startup.append(_start_live_view)
            app.on_cleanup.append(_stop_live_view)
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static/", settings.web_dir / "phone", show_index=False)
    app.on_shutdown.append(_close_sessions)
    return app


def session_status(app: web.Application) -> dict[str, object]:
    """Snapshot of the phone app for the dashboard: active session and reconstruction jobs."""
    sessions = list(app[SESSIONS_KEY].values())
    jobs = app[JOBS_KEY].snapshot() if JOBS_KEY in app else []
    status: dict[str, object] = {
        "session": sessions[0].stats() if sessions else None,
        "jobs": jobs,
    }
    if CALIBRATION_KEY in app:
        calibration = app[CALIBRATION_KEY]
        status["calibration"] = calibration.snapshot()
        if sessions:
            status["focal_ratio"] = calibration.saved_focal_ratio(_session_device(sessions[0]))
    if DEMO_KEY in app:
        status["demo"] = app[DEMO_KEY].snapshot()
    return status


class RecordButton:
    """The dashboard's view of the phone app: start/stop clips, list jobs."""

    def __init__(self, phone_app: web.Application) -> None:
        self._app = phone_app

    async def set(self, on: bool) -> dict[str, object]:
        """Start or stop a clip on the active session.

        Raises:
            web.HTTPConflict: If no phone is connected or recording is unavailable.
        """
        sessions = list(self._app[SESSIONS_KEY].values())
        if not sessions:
            raise web.HTTPConflict(
                text="No phone is connected. Start streaming on the phone first."
            )
        if CONTROL_KEY not in self._app:
            raise web.HTTPConflict(text="Recording is not available on this server.")
        control, session = self._app[CONTROL_KEY], sessions[0]
        if on:
            await control.start(session)
        else:
            await control.stop(session)
        return {"clip": session.stats()["clip"]}

    async def calibrate(self, on: bool) -> dict[str, object]:
        """Start or stop live camera calibration on the active session.

        Raises:
            web.HTTPConflict: If no phone is connected, or calibration cannot
                start (marker size not set, OpenCV missing).
        """
        if CALIBRATION_KEY not in self._app:
            raise web.HTTPConflict(text="Calibration is not available on this server.")
        calibration = self._app[CALIBRATION_KEY]
        if not on:
            await asyncio.to_thread(calibration.stop)
            return calibration.snapshot()
        sessions = list(self._app[SESSIONS_KEY].values())
        if not sessions:
            raise web.HTTPConflict(
                text="No phone is connected. Start streaming on the phone first."
            )
        try:
            calibration.start(_session_device(sessions[0]))
        except RuntimeError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        return calibration.snapshot()

    async def demo(self, on: bool, recalibrate: bool = False) -> dict[str, object]:
        """Start or stop the distance demo on the active session.

        Raises:
            web.HTTPConflict: If no phone is connected, or the demo cannot
                start (marker size not set, OpenCV missing).
        """
        if DEMO_KEY not in self._app:
            raise web.HTTPConflict(text="The distance demo is not available on this server.")
        demo = self._app[DEMO_KEY]
        if not on:
            await asyncio.to_thread(demo.stop)
            return demo.snapshot()
        sessions = list(self._app[SESSIONS_KEY].values())
        if not sessions:
            raise web.HTTPConflict(
                text="No phone is connected. Start streaming on the phone first."
            )
        try:
            await asyncio.to_thread(demo.start, _session_device(sessions[0]), recalibrate)
        except RuntimeError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        return demo.snapshot()

    def jobs(self) -> list[dict[str, object]]:
        """Reconstruction jobs, newest first."""
        return self._app[JOBS_KEY].snapshot() if JOBS_KEY in self._app else []


def create_ssl_context(settings: Settings) -> ssl.SSLContext | None:
    """Return a server TLS context, or None when TLS is disabled."""
    if settings.tls_cert is None or settings.tls_key is None:
        return None
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(settings.tls_cert, settings.tls_key)
    return context
