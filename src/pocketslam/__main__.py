"""Entry point: ``uv run python -m pocketslam`` or ``uv run pocketslam``.

Runs two web apps in one process:

* the phone-facing HTTPS server (camera + IMU in, WebRTC signaling), and
* the laptop dashboard on localhost (live view and recording status).
"""

from __future__ import annotations

import asyncio
import logging
import signal

from aiohttp import web

from pocketslam.calib.demo import DistanceDemo
from pocketslam.calib.distance import LiveDistance
from pocketslam.calib.live import LiveCalibration
from pocketslam.calib.store import ScaleCalibration
from pocketslam.config import ReconSettings, Settings
from pocketslam.core.device import select_device
from pocketslam.log import configure_logging
from pocketslam.record.telemetry import Telemetry
from pocketslam.server.app import (
    RecordButton,
    create_app,
    create_ssl_context,
    phone_calibration_notifier,
    phone_distance_notifier,
    session_status,
)
from pocketslam.server.dashboard import (
    LiveHub,
    create_dashboard_app,
    phone_url,
    start_on_first_free_port,
)
from pocketslam.server.recording import ReconJobs

logger = logging.getLogger("pocketslam")

SHUTDOWN_TIMEOUT_S = 10.0


async def serve(settings: Settings) -> None:
    """Start both apps and run until SIGINT or SIGTERM."""
    ssl_context = create_ssl_context(settings)
    if ssl_context is None:
        logger.warning("serving plain HTTP: only a browser on this machine can use the camera")

    hub = LiveHub()
    jobs = ReconJobs(settings) if settings.auto_reconstruct else None
    recon = ReconSettings()
    calibration = LiveCalibration(
        ScaleCalibration(recon.calibration_file),
        recon.marker_size_m,
        recon.marker_id,
        recon.marker_dictionary,
    )
    distance = LiveDistance(recon.marker_size_m, recon.marker_id, recon.marker_dictionary)
    demo = DistanceDemo(calibration, distance)
    phone_app = create_app(
        settings, Telemetry(), live=hub, jobs=jobs, calibration=calibration, demo=demo
    )
    calibration_to_phone = phone_calibration_notifier(phone_app)
    distance_to_phone = phone_distance_notifier(phone_app)

    def on_calibration_update(snapshot: dict[str, object]) -> None:
        hub.publish_json({"type": "calibration", **snapshot})
        calibration_to_phone(snapshot)

    def on_distance_update(snapshot: dict[str, object]) -> None:
        hub.publish_json({"type": "distance", "stage": demo.stage, **snapshot})
        distance_to_phone(snapshot)

    calibration.on_update = on_calibration_update
    distance.on_update = on_distance_update
    url = phone_url(settings)
    dashboard_app = create_dashboard_app(
        settings,
        hub,
        status=lambda: {
            "phone_url": url,
            "recording_enabled": settings.record,
            **session_status(phone_app),
        },
        record_button=RecordButton(phone_app),
    )

    phone_runner = web.AppRunner(phone_app, shutdown_timeout=SHUTDOWN_TIMEOUT_S, access_log=None)
    dashboard_runner = web.AppRunner(dashboard_app, shutdown_timeout=2.0, access_log=None)
    await phone_runner.setup()
    await dashboard_runner.setup()
    try:
        await web.TCPSite(
            phone_runner, settings.host, settings.port, ssl_context=ssl_context
        ).start()
        dash_port = await start_on_first_free_port(
            dashboard_runner, settings.dashboard_host, settings.dashboard_ports
        )
        logger.info("phone page", extra={"url": url})
        logger.info("dashboard", extra={"url": f"http://{settings.dashboard_host}:{dash_port}"})

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        logger.info("shutting down")
    finally:
        # Dashboard first so no tab holds the loop; then the phone app closes
        # sessions and finishes MCAP files.
        await dashboard_runner.cleanup()
        await phone_runner.cleanup()


def main() -> None:
    """Load settings, check the compute device, and serve."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    select_device(settings.device)
    asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
