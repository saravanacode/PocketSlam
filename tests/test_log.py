import json
import logging

from pocketslam.log import JsonFormatter, configure_logging


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord(
        "pocketslam.x", logging.INFO, __file__, 1, "hello %s", ("phone",), None
    )
    record.session_id = "abc"
    out = json.loads(JsonFormatter().format(record))
    assert out["msg"] == "hello phone"
    assert out["level"] == "INFO"
    assert out["session_id"] == "abc"


def test_configure_logging_replaces_handlers() -> None:
    configure_logging("WARNING", "text")
    configure_logging("DEBUG", "json")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.DEBUG
