"""Runtime settings. Every value comes from ``POCKETSLAM_*`` env vars or ``config/.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """PocketSlam settings.

    Attributes:
        host: Interface the phone-facing HTTPS server binds to. ``0.0.0.0``
            lets phones on the LAN connect.
        port: Phone-facing HTTPS server port.
        dashboard_host: Interface the laptop dashboard binds to.
        dashboard_ports: Dashboard ports to try in order; the first free one wins.
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
        record: Also write every whole phone session to an MCAP file. Off by
            default: the Record button saves clips instead.
        auto_reconstruct: Run ``pocketslam-recon`` on every saved clip.
        recordings_dir: Where MCAP recordings are written.
        reconstructions_dir: Where ``pocketslam-recon`` writes scans; the
            dashboard's 3D tab lists this directory.
        jpeg_quality: JPEG quality for recorded and live camera frames.
        record_queue_size: Max items waiting for the recording thread.
        live_view: Also serve live data to the Foxglove app over WebSocket.
        live_view_host: Interface the Foxglove WebSocket server binds to.
        live_view_port: Foxglove WebSocket server port.
        log_format: ``json`` for production, ``text`` for local reading.
        log_level: Root log level.
    """

    model_config = SettingsConfigDict(
        env_prefix="POCKETSLAM_", env_file="config/.env", extra="ignore"
    )

    host: str = "0.0.0.0"  # noqa: S104 - phones on the LAN must reach it; TLS is required
    port: int = Field(default=10001, ge=1, le=65535)
    dashboard_host: str = "127.0.0.1"
    dashboard_ports: list[int] = Field(default=[8000, 8001], min_length=1)
    tls_cert: Path | None = None
    tls_key: Path | None = None
    allow_insecure: bool = False
    device: str | None = None
    default_hfov_deg: float = Field(default=65.0, gt=0.0, lt=180.0)
    queue_size: int = Field(default=2, ge=1)
    max_message_bytes: int = Field(default=16_384, ge=256)
    web_dir: Path = Path("web")
    output_dir: Path = Path("data/outputs")
    record: bool = False
    auto_reconstruct: bool = True
    recordings_dir: Path = Path("data/recordings")
    reconstructions_dir: Path = Path("data/reconstructions")
    jpeg_quality: int = Field(default=85, ge=1, le=95)
    record_queue_size: int = Field(default=120, ge=1)
    live_view: bool = False
    live_view_host: str = "127.0.0.1"
    live_view_port: int = Field(default=8765, ge=0, le=65535)
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


class ReconSettings(BaseSettings):
    """Settings for offline reconstruction (``pocketslam-recon``).

    Model ids are pinned to Hugging Face revisions (RULES.md 5.2). Change the
    id and revision together.

    Attributes:
        model: Any-view Depth Anything 3 checkpoint (depth + poses + intrinsics).
        model_revision: Pinned commit of ``model``.
        metric_model: Metric monocular checkpoint for the scale cross-check.
        metric_revision: Pinned commit of ``metric_model``.
        use_metric_check: Run the metric model to check or replace the IMU scale.
        device: Compute backend override; None picks automatically.
        process_res: Longest image side the model processes, in pixels.
        max_frames: Most keyframes sent to the model.
        keyframe_fps: Target keyframe rate before the ``max_frames`` cap.
        conf_percentile: Drop pixels below this per-frame confidence percentile.
        voxel_m: Voxel size for point-cloud downsampling, metres.
        output_dir: Where reconstructions are written.
        marker_size_m: Edge of the fiducial marker's black square in metres.
            None turns marker scaling off.
        marker_id: Only use this marker id; None accepts any.
        marker_dictionary: OpenCV ArUco dictionary name.
        use_calibration: Apply the remembered marker correction to scans
            without a marker.
        use_calibrated_intrinsics: Use the focal length measured with
            ``pocketslam-calibrate`` in place of the model's own estimate.
        calibration_file: Where that correction is stored.
    """

    model_config = SettingsConfigDict(
        env_prefix="POCKETSLAM_RECON_", env_file="config/.env", extra="ignore"
    )

    model: str = "depth-anything/DA3-LARGE-1.1"
    model_revision: str = "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"
    metric_model: str = "depth-anything/DA3METRIC-LARGE"
    metric_revision: str = "4010e39f3634a45bc60553321fb49fb760bd594e"
    use_metric_check: bool = True
    device: str | None = None
    process_res: int = Field(default=504, ge=112)
    max_frames: int = Field(default=48, ge=2)
    keyframe_fps: float = Field(default=10.0, gt=0.0)
    conf_percentile: float = Field(default=40.0, ge=0.0, lt=100.0)
    voxel_m: float = Field(default=0.01, gt=0.0)
    output_dir: Path = Path("data/reconstructions")
    marker_size_m: float | None = Field(default=None, gt=0.0)
    marker_id: int | None = Field(default=None, ge=0)
    marker_dictionary: str = "DICT_4X4_50"
    use_calibration: bool = True
    use_calibrated_intrinsics: bool = True
    calibration_file: Path = Path("data/calibration/scale.json")
