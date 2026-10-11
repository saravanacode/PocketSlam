"""JSON that browsers can parse.

Python's ``json`` writes ``NaN`` and ``Infinity`` for non-finite floats, which
is not valid JSON: ``JSON.parse`` in a browser rejects the whole document.
Everything sent to the dashboard or written for it goes through here.
"""

from __future__ import annotations

import json
import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Copy of ``value`` with every non-finite float replaced by None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    """``json.dumps`` that never emits ``NaN`` or ``Infinity`` (they become ``null``)."""
    return json.dumps(json_safe(value), allow_nan=False, default=str, **kwargs)
