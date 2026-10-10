"""Container health probe: ``python -m pocketslam.healthcheck``.

Exits 0 when ``/healthz`` answers 200, 1 otherwise. Reads the same settings as
the server, so it follows the configured port and HTTP/HTTPS choice.
"""

from __future__ import annotations

import ssl
import sys
import urllib.request
from urllib.error import URLError

from pocketslam.config import Settings

TIMEOUT_S = 2.0
OK = 200


def healthz_url(settings: Settings) -> str:
    """Return the loopback URL of the server's ``/healthz`` endpoint."""
    scheme = "https" if settings.tls_enabled else "http"
    host = "127.0.0.1" if settings.host in {"0.0.0.0", "::"} else settings.host  # noqa: S104 - comparing, not binding
    return f"{scheme}://{host}:{settings.port}/healthz"


def check(url: str) -> bool:
    """Return True if ``url`` answers HTTP 200.

    Certificate verification is off: this probes our own server over
    loopback, and dev certificates are not in the container's trust store.
    """
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S, context=context) as resp:  # noqa: S310 - URL built from settings, http(s) only
            return bool(resp.status == OK)
    except (URLError, OSError):
        return False


def main() -> int:
    """Probe the server and return a process exit code."""
    return 0 if check(healthz_url(Settings())) else 1


if __name__ == "__main__":
    sys.exit(main())
