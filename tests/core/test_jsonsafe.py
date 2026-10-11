import json

import pytest

from pocketslam.core.jsonsafe import dumps, json_safe


def test_non_finite_floats_become_null() -> None:
    data = {"a": float("nan"), "b": [1.5, float("inf"), {"c": -float("inf")}], "d": (2, "x")}
    assert json_safe(data) == {"a": None, "b": [1.5, None, {"c": None}], "d": [2, "x"]}


def test_dumps_output_is_strict_json() -> None:
    text = dumps({"path_length_m": float("nan"), "points": 1})
    assert "NaN" not in text

    def reject(token: str) -> None:
        raise AssertionError(token)

    assert json.loads(text, parse_constant=reject) == {"path_length_m": None, "points": 1}


def test_plain_json_dumps_would_have_been_invalid() -> None:
    # The bug this module prevents: Python emits NaN, which browsers reject.
    assert "NaN" in json.dumps({"x": float("nan")})
    with pytest.raises(ValueError, match="Out of range"):
        json.dumps({"x": float("nan")}, allow_nan=False)
