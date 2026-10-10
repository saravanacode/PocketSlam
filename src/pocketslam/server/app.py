"""HTTPS server: serves the phone client and handles WebRTC signaling."""

from __future__ import annotations

import asyncio
import logging
import ssl
from typing import Any

from aiohttp import web

from pocketslam.config import Settings
from pocketslam.pipeline import Session, start_session

logger = logging.getLogger(__name__)

SETTINGS_KEY = web.AppKey("settings", Settings)
SESSIONS_KEY = web.AppKey("sessions", dict[str, Session])
MAX_OFFER_BYTES = 64 * 1024


async def healthz(_request: web.Request) -> web.Response:
    """Liveness probe used by Docker HEALTHCHECK."""
    return web.json_response({"status": "ok"})


async def index(request: web.Request) -> web.FileResponse:
    """Serve the phone client page."""
    return web.FileResponse(request.app[SETTINGS_KEY].web_dir / "index.html")


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

    def _forget(session: Session) -> None:
        sessions.pop(session.session_id, None)
        asyncio.get_running_loop().create_task(session.close())

    session, peer = await start_session(
        request.app[SETTINGS_KEY], body["sdp"], body["type"], on_closed=_forget
    )
    sessions[session.session_id] = session
    logger.info("session started", extra={"session_id": session.session_id})
    return web.json_response(
        {"sdp": peer.answer.sdp, "type": peer.answer.type, "session_id": session.session_id}
    )


async def _close_sessions(app: web.Application) -> None:
    sessions = list(app[SESSIONS_KEY].values())
    app[SESSIONS_KEY].clear()
    await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)


def create_app(settings: Settings) -> web.Application:
    """Build the aiohttp application.

    Args:
        settings: Runtime settings.

    Returns:
        The configured application.
    """
    app = web.Application(client_max_size=MAX_OFFER_BYTES)
    app[SETTINGS_KEY] = settings
    app[SESSIONS_KEY] = {}
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static/", settings.web_dir, show_index=False)
    app.on_shutdown.append(_close_sessions)
    return app


def create_ssl_context(settings: Settings) -> ssl.SSLContext | None:
    """Return a server TLS context, or None when TLS is disabled."""
    if settings.tls_cert is None or settings.tls_key is None:
        return None
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(settings.tls_cert, settings.tls_key)
    return context
