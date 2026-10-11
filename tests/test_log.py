import json
import logging
from collections.abc import Iterator

import pytest

from pocketslam.log import JsonFormatter, SessionLogger, TextFormatter, configure_logging


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord(
        "pocketslam.x", logging.INFO, __file__, 1, "hello %s", ("phone",), None
    )
    record.session_id = "abc"
    out = json.loads(JsonFormatter().format(record))
    assert out["msg"] == "hello phone"
    assert out["level"] == "INFO"
    assert out["session_id"] == "abc"


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    """configure_logging binds a handler to this test's captured stderr; undo it."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_replaces_handlers() -> None:
    configure_logging("WARNING", "text")
    configure_logging("DEBUG", "json")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.DEBUG


def test_text_formatter_appends_extra_fields() -> None:
    record = logging.LogRecord("pocketslam", logging.INFO, __file__, 1, "dashboard", (), None)
    record.url = "http://127.0.0.1:8000"
    assert TextFormatter().format(record).endswith("dashboard  url=http://127.0.0.1:8000")


def test_session_logger_keeps_call_fields(caplog: pytest.LogCaptureFixture) -> None:
    log = SessionLogger(logging.getLogger("pocketslam.test"), {"session_id": "abc"})
    with caplog.at_level(logging.INFO, logger="pocketslam.test"):
        log.info("link stats", extra={"recv_fps": 30.0})
    record = caplog.records[-1]
    assert (record.session_id, record.recv_fps) == ("abc", 30.0)
