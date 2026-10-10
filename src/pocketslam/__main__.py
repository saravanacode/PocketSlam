"""Entry point: ``uv run python -m pocketslam`` or ``uv run pocketslam``."""

from __future__ import annotations

import logging

from aiohttp import web

from pocketslam.config import Settings
from pocketslam.core.device import select_device
from pocketslam.log import configure_logging
from pocketslam.server.app import create_app, create_ssl_context

logger = logging.getLogger("pocketslam")

SHUTDOWN_TIMEOUT_S = 10.0


def main() -> None:
    """Load settings, check the compute device, and serve until SIGTERM/SIGINT."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    select_device(settings.device)
    ssl_context = create_ssl_context(settings)
    scheme = "https" if ssl_context else "http"
    if ssl_context is None:
        logger.warning("serving plain HTTP: only a browser on this machine can use the camera")
    logger.info("listening", extra={"url": f"{scheme}://{settings.host}:{settings.port}"})
    # run_app handles SIGINT/SIGTERM and runs on_shutdown hooks, which close sessions.
    web.run_app(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        ssl_context=ssl_context,
        shutdown_timeout=SHUTDOWN_TIMEOUT_S,
        print=None,
        access_log=None,
    )


if __name__ == "__main__":
    main()
