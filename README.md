# PocketSlam

Device-agnostic monocular depth and SLAM. Any phone streams its camera over WebRTC to a Mac (or Linux box), which runs monocular depth estimation and SLAM.

The phone needs only a browser. The host picks CUDA, MPS, or CPU automatically.

> Status: skeleton. Transport, session pipeline, and phone client work end to end. Depth (`ConstantDepthEstimator`) and SLAM (`StaticSlam`) are placeholders.

## Quick start (macOS, native)

Run natively on a Mac. Docker on macOS cannot use the Apple GPU.

```bash
uv sync
brew install mkcert
scripts/gen_dev_cert.sh          # writes data/certs/, prints the config/.env lines to add
cp config/env.example config/.env  # then edit if needed
uv run python -m pocketslam
```

Open `https://<mac-lan-ip>:8443` on the phone and tap **Start streaming**. The phone must trust the mkcert root CA (see the comments in `scripts/gen_dev_cert.sh`).

For a browser on the Mac itself, HTTPS is not required:

```bash
POCKETSLAM_ALLOW_INSECURE=1 POCKETSLAM_LOG_FORMAT=text uv run python -m pocketslam
```

Trajectories are written to `data/outputs/<session-id>/trajectory.tum`.

## Docker (Linux hosts)

```bash
scripts/gen_dev_cert.sh
docker compose up --build
```

`compose.yaml` uses host networking because WebRTC needs UDP ports that cannot be published reliably.

## Development

All rules are in [RULES.md](RULES.md). Before you push:

```bash
uv run pre-commit install --config config/pre-commit.yaml  # once
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports --cache-dir .cache/import_linter
uv run scripts/check_structure.py
uv run pytest
```
