"""Runtime settings. Every value comes from ``POCKETSLAM_*`` env vars or ``config/.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """PocketSlam settings.

    Attributes:
        host: Interface the HTTPS server binds to.
        port: HTTPS server port.
        tls_cert: PEM certificate path. Phones need HTTPS for camera access.
        tls_key: PEM private key path.
        allow_insecure: Serve plain HTTP when no certificate is set. Only
            useful for a browser on the same machine (``localhost``).
        device: Compute backend override, validated by ``core.device``;
            None picks automatically.
        default_hfov_deg: Horizontal FOV used when a phone sends no intrinsics.
        queue_size: Max frames buffered between transport and pipeline.
        max_message_bytes: Max DataChannel message size accepted.
        web_dir: Directory with the phone client.
        output_dir: Where session outputs are written.
        log_format: ``json`` for production, ``text`` for local reading.
        log_level: Root log level.
    """

    model_config = SettingsConfigDict(
        env_prefix="POCKETSLAM_", env_file="config/.env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8443, ge=1, le=65535)
    tls_cert: Path | None = None
    tls_key: Path | None = None
    allow_insecure: bool = False
    device: str | None = None
    default_hfov_deg: float = Field(default=65.0, gt=0.0, lt=180.0)
    queue_size: int = Field(default=2, ge=1)
    max_message_bytes: int = Field(default=16_384, ge=256)
    web_dir: Path = Path("web")
    output_dir: Path = Path("data/outputs")
    log_format: Literal["json", "text"] = "json"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _check_tls(self) -> Settings:
        if (self.tls_cert is None) != (self.tls_key is None):
            msg = "set both POCKETSLAM_TLS_CERT and POCKETSLAM_TLS_KEY, or neither"
            raise ValueError(msg)
        if self.tls_cert is None and not self.allow_insecure:
            msg = (
                "no TLS certificate set. Phones need HTTPS for camera access. "
                "Run scripts/gen_dev_cert.sh, or set POCKETSLAM_ALLOW_INSECURE=1 "
                "for a localhost-only browser."
            )
            raise ValueError(msg)
        return self

    @property
    def tls_enabled(self) -> bool:
        """True when the server serves HTTPS."""
        return self.tls_cert is not None
