"""Logging setup: JSON lines in production, plain text for local runs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Attributes every LogRecord has, plus those Formatter.format() adds. Anything
# else came from ``extra=`` and is copied into the output.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON, including fields passed via ``extra=``."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable lines with fields from ``extra=`` appended as ``key=value``."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as text, then append its extra fields."""
        line = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _RESERVED}
        if extras:
            line += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        return line


class SessionLogger(logging.LoggerAdapter[logging.Logger]):
    """Adds fixed fields (such as ``session_id``) to every record.

    The stock ``LoggerAdapter`` replaces a call's ``extra=`` with its own on
    Python 3.12, which silently dropped per-call fields. This one merges them.
    """

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        """Merge the adapter's fields under the call's ``extra``."""
        kwargs["extra"] = {**(self.extra or {}), **kwargs.get("extra", {})}
        return msg, kwargs


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root logger.

    Args:
        level: Root log level name.
        fmt: ``"json"`` or ``"text"``.
    """
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
