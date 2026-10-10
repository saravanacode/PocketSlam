from pathlib import Path

import pytest
from pydantic import ValidationError

from pocketslam.config import Settings


def test_requires_tls_unless_insecure_allowed() -> None:
    with pytest.raises(ValidationError, match="no TLS certificate"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_cert_and_key_must_be_set_together() -> None:
    with pytest.raises(ValidationError, match="both"):
        Settings(tls_cert=Path("cert.pem"), _env_file=None)  # type: ignore[call-arg]


def test_reads_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POCKETSLAM_ALLOW_INSECURE", "1")
    monkeypatch.setenv("POCKETSLAM_PORT", "9000")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.port == 9000
    assert not s.tls_enabled
