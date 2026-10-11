# PocketSlam Engineering Rules

These rules apply to all code in this repository. CI enforces every rule marked **[gate]**. A pull request that fails any gate does not merge.

## 1. What the system does

```
 Phone (Chrome on Android)                   Laptop (Mac / Linux)
 ┌─────────────────────────┐  WebRTC video   ┌───────────────────────────────────────────┐
 │ web/phone/  :10001 HTTPS│ ──────────────▶ │ transport/  frames, metadata, IMU (aiortc)│
 │ rear camera             │  DataChannel    │      │                                    │
 │ gyroscope, accelerometer│ ──────────────▶ │      ├──▶ record/   MCAP file + live data │
 └─────────────────────────┘                 │      │       │                            │
          ▲  HTTPS signaling (offer/answer)  │      │       ▼                            │
          └───────────────────────────────── │      │    server/dashboard  :8000 (8001)  │
                                             │      ▼    web/dashboard/ live view        │
                                             │ depth/ ─▶ slam/ ─▶ viz/                   │
                                             └───────────────────────────────────────────┘
```

"Device-agnostic" means two things, and both are hard requirements:

1. **Capture side:** any phone with a modern browser works. There is no native app. The phone opens a web page served by the host.
2. **Compute side:** the same code runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU. No code path assumes one backend.

## 2. Toolchain

- Python is managed by **uv** only. The project has `pyproject.toml`, `uv.lock`, and `.python-version`.
- Python version is pinned to **3.12** (`uv python pin 3.12`). Do not move to a newer version until PyTorch, aiortc, and the depth models ship wheels for it.
- Add dependencies with `uv add <pkg>`. Add dev tools with `uv add --dev <pkg>`. Never use `pip install` or `uv pip install`.
- Run everything with `uv run ...`. Docs, scripts, and CI never call `python`, `pip`, or `.venv/bin/...` directly.
- `uv.lock` is committed. CI runs `uv sync --frozen`, so an out-of-date lock file fails the build.
- On Linux, `torch` resolves from the PyTorch CPU index (`[tool.uv.sources]` in `pyproject.toml`). The default Linux wheels on PyPI bundle CUDA and add several GB to the image. macOS uses PyPI wheels, which include MPS.

## 3. Quality gates

Run all gates locally before you push:

```bash
uv run ruff check .                      # lint            [gate]
uv run ruff format --check .             # formatting      [gate]
uv run mypy src                          # types           [gate]
uv run lint-imports --cache-dir .cache/import_linter  # layer contracts [gate]
uv run scripts/check_structure.py        # repo layout     [gate]
uv run pytest -m "not gpu and not slow"  # tests           [gate]
docker run --rm -i hadolint/hadolint:v2.12.0 < docker/Dockerfile  # Dockerfile lint [gate]
docker build -f docker/Dockerfile .      # image builds    [gate]
```

`pre-commit` runs the fast gates (ruff, ruff format, check_structure, `uv lock --check`) on every commit. Install it once with `uv run pre-commit install --config config/pre-commit.yaml`.

### 3.1 Ruff [gate]

Ruff is the only linter and formatter. Do not add black, isort, flake8, or pylint.

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = [
  "E", "W",   # pycodestyle
  "F",        # pyflakes
  "I",        # isort
  "B",        # bugbear
  "UP",       # pyupgrade
  "SIM",      # simplify
  "N",        # naming
  "C4",       # comprehensions
  "PTH",      # use pathlib
  "RUF",      # ruff-specific
  "ASYNC",    # async pitfalls (we run an asyncio server)
  "T20",      # no print()
  "S",        # bandit security
  "PL",       # pylint subset
  "NPY",      # numpy
  "PERF",     # performance
  "D",        # docstrings
]
ignore = ["D100", "D104", "D105", "D107", "PLR0913"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "D", "PLR2004"]
"scripts/**" = ["T20", "D"]
```

Rules:

- `# noqa` must name the rule and give a reason: `# noqa: S104 - bind all interfaces for LAN phones`.
- Do not add a global `ignore` entry without a comment in `pyproject.toml` that explains why.

### 3.2 Types [gate]

- `mypy --strict` on `src/`. Every public function has full type hints.
- Arrays use `numpy.typing.NDArray[np.float32]` (or the right dtype), never bare `np.ndarray`.
- Tensors document shape and dtype in the docstring, for example `(H, W) float32 metres`.
- `# type: ignore` must name the error code: `# type: ignore[import-untyped]`.

### 3.3 Tests [gate]

- `pytest` with `pytest-asyncio`. Tests live in `tests/` and mirror the `src/pocketslam/` layout.
- Unit tests need no network, no camera, no GPU, and no downloaded model weights. Use small synthetic frames and fake depth models.
- Tests that need a GPU are marked `@pytest.mark.gpu`. Tests that take more than 5 s are marked `@pytest.mark.slow`. CI runs these in a separate job.
- Every bug fix adds a test that fails before the fix.
- SLAM changes include a regression run on a recorded sequence (see section 8). Report ATE/RPE before and after in the PR.

## 4. Project structure

### 4.1 Layout [gate]

```
PocketSlam/
├── pyproject.toml
├── uv.lock
├── .python-version
├── RULES.md
├── README.md
├── .gitignore
├── compose.yaml
├── .github/
│   ├── workflows/ci.yml        # runs every gate in section 3
│   └── dependabot.yml          # bumps uv deps, image digests, actions
├── docker/
│   ├── Dockerfile
│   └── Dockerfile.dockerignore # BuildKit reads it next to the Dockerfile
├── config/                     # non-Python config, kept out of the root
│   ├── pre-commit.yaml
│   ├── env.example             # copy to config/.env (never committed)
│   └── .env
├── scripts/
│   ├── check_structure.py      # enforces this section
│   └── gen_dev_cert.sh         # local TLS cert for phone HTTPS
├── web/                        # static HTML/JS/CSS, no build step
│   ├── phone/                  # capture page, served on :10001 over HTTPS
│   └── dashboard/              # laptop live view, served on localhost :8000
├── src/pocketslam/
│   ├── __init__.py
│   ├── __main__.py             # entry point: `uv run python -m pocketslam`
│   ├── config.py               # pydantic-settings, env vars only
│   ├── log.py                  # JSON / text logging setup
│   ├── healthcheck.py          # container probe: python -m pocketslam.healthcheck
│   ├── core/                   # shared types, no heavy dependencies
│   │   ├── types.py            # Frame, Intrinsics, Pose, DepthMap
│   │   └── device.py           # the ONLY place that picks cuda/mps/cpu
│   ├── server/                 # phone HTTPS + signaling, dashboard + LiveHub (aiohttp)
│   ├── record/                 # Foxglove channels, MCAP writer, recording thread
│   ├── recon/                  # offline reconstruction: reader, fuse, export, CLI
│   ├── transport/              # WebRTC receive, frame queue, DataChannel
│   ├── depth/                  # depth model interface + backends
│   ├── slam/                   # tracking, mapping, loop closure
│   ├── viz/                    # visualisation
│   └── pipeline.py             # wires transport → depth → slam → viz
├── tests/
└── data/                       # gitignored: recordings, weights, outputs
```

### 4.2 Clean root [gate]

- Tool caches go in `.cache/<tool>/`. Set the path in `pyproject.toml` (ruff `cache-dir`, mypy `cache_dir`, pytest `cache_dir`). import-linter has no config option for this, so always run it as `lint-imports --cache-dir .cache/import_linter`.
- When you add a tool, point its cache or config into `.cache/` or `config/` before you commit.

### 4.3 Layer contracts [gate]

Dependencies point one way only. `import-linter` enforces this.

```
server ─▶ pipeline ─▶ transport ─┐
                    ├▶ record   ─┤
                    ├▶ depth    ─┼─▶ core
                    ├▶ slam     ─┤
                    └▶ viz      ─┘

recon (offline) ─▶ depth, slam, record ─▶ core   (never imported by the live path)
```

```toml
[tool.importlinter]
root_package = "pocketslam"

[[tool.importlinter.contracts]]
name = "core depends on nothing inside pocketslam"
type = "forbidden"
source_modules = ["pocketslam.core"]
forbidden_modules = [
  "pocketslam.server", "pocketslam.transport", "pocketslam.depth",
  "pocketslam.slam", "pocketslam.viz", "pocketslam.pipeline",
]

[[tool.importlinter.contracts]]
name = "processing modules are independent"
type = "independence"
modules = ["pocketslam.transport", "pocketslam.depth", "pocketslam.slam", "pocketslam.viz"]

[[tool.importlinter.contracts]]
name = "nothing imports the server"
type = "forbidden"
source_modules = [
  "pocketslam.core", "pocketslam.transport", "pocketslam.depth",
  "pocketslam.slam", "pocketslam.viz", "pocketslam.pipeline",
]
forbidden_modules = ["pocketslam.server"]
```

Consequences:

- `depth/` and `slam/` do not know WebRTC exists. They take `core.types.Frame` and return `core.types` objects. You can run them on a video file with no network.
- `slam/` does not import a specific depth model. It takes a `DepthMap`.
- `record/` does not know about the dashboard. It pushes encoded frames and IMU to a `LiveSink` protocol, which `server/dashboard.py` implements.

### 4.4 `scripts/check_structure.py` [gate]

This script exits non-zero if any of these checks fail:

1. Every required file and directory in section 4.1 exists.
2. Every package directory under `src/pocketslam/` has an `__init__.py`.
3. Every module `src/pocketslam/<pkg>/<mod>.py` has a test file `tests/<pkg>/test_<mod>.py`.
4. The strings `.cuda(`, `"cuda"`, `"mps"`, and `torch.device(` appear only in `src/pocketslam/core/device.py`.
5. No files over 5 MB are tracked by git. Weights and recordings go in `data/` or are downloaded.
6. No `requirements*.txt`, `setup.py`, or `setup.cfg` files exist. `pyproject.toml` is the only source of truth.
7. `.python-version` matches `requires-python` in `pyproject.toml`.
8. No `print(` appears in `src/` (ruff `T20` also checks this; the script is a backstop).
9. The repo root has no hidden entries except `.git`, `.github`, `.venv`, `.cache`, `.gitignore`, and `.python-version`. Git, GitHub, and uv need these in the root. Everything else goes under `config/` or `docker/`, and every tool cache goes under `.cache/`.

## 5. Device-agnostic rules

### 5.1 Compute backend

- `core/device.py` exposes `select_device(preference: str | None) -> torch.device`. The order is `cuda`, then `mps`, then `cpu`. The `POCKETSLAM_DEVICE` env var overrides it.
- All other code receives a device as an argument. No module picks a device by itself.
- Ops that MPS does not support must have a CPU fallback in the same function. Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1` globally to hide them; it silently moves work to CPU and kills frame rate.
- Use `float32` by default. Use `float16`/`bfloat16` only behind a config flag, and only after a test shows the same depth error on all three backends.

### 5.2 Depth models

- Every backend implements one protocol:

  ```python
  class DepthEstimator(Protocol):
      def estimate(self, frame: Frame) -> DepthMap: ...
  ```

- `DepthMap` states whether depth is **metric** (metres) or **relative** (affine-invariant). SLAM must check this flag and never treat relative depth as metric.
- Model weights are pinned by repo + revision + sha256 in config. Code never downloads "latest".
- Weights are never committed and never baked into the image. They are mounted at `/models` or downloaded at startup into a cache volume.

### 5.3 Phone capture

- The client is plain HTML/JS/CSS in `web/phone/`. Android Chrome is the primary target; iOS Safari must not break. Test on a real phone before you merge client changes.
- `getUserMedia` and motion sensors need a secure context, so the host serves the client over **HTTPS** on the LAN. `scripts/gen_dev_cert.sh` uses mkcert if installed, otherwise a self-signed certificate. Never disable TLS to make a phone connect.
- Ask for the rear camera with `facingMode: "environment"` and an ideal resolution. Accept what the browser actually gives and report the real width and height to the host.
- Each phone sends session metadata on the DataChannel: actual resolution, orientation, and intrinsics if known. If intrinsics are unknown, the host uses a calibration file for that device or an estimated value, and logs which one it used.
- IMU is optional; the pipeline must run without it. The client uses the Generic Sensor API (`Gyroscope`, `Accelerometer`) at 100 Hz and falls back to `devicemotion` events if a sensor cannot start. iOS needs a permission tap.
- IMU units are fixed: gyroscope in rad/s, accelerometer in m/s² **including gravity**, both in the phone's device frame. `devicemotion` reports deg/s with alpha/beta/gamma about z/x/y; the client converts it.
- IMU travels as batched DataChannel messages: `{"type": "imu", "samples": [["g"|"a", t_ms, x, y, z], ...]}`, where `t_ms` is the phone's Unix time in ms. `transport/webrtc.py` validates every message.
- Every sample has a sender timestamp: RTP time for video, sensor time for IMU. `core/clock.ClockMapper` maps each stream onto the host clock with a minimum-offset estimate. Never use host arrival time directly; it carries network jitter.

### 5.4 Real-time behaviour

- Queues between stages are bounded. When a queue is full, drop the **oldest** frame. Latency beats completeness for live tracking.
- Never block the asyncio event loop. Model inference and SLAM run in a worker thread or process.
- Log per-stage latency and FPS. A PR that makes end-to-end latency worse by more than 10% on the reference sequence must explain why.

### 5.5 Recording and live view

- Clips are recorded with the **Record** button (phone page or dashboard strip). A clip goes to `data/recordings/clip_<UTC time>.mcap` and holds exactly the frames and IMU submitted between start and stop: start and stop travel through the recording thread's queue (`record/worker.py`). A clip still running when the phone disconnects is saved, not lost.
- The phone sends `{"type": "record", "on": true|false}`; the laptop answers `{"type": "record_state", "on": ..., "clip": ..., "saved": ...}`. The dashboard uses `POST /api/record`.
- Every saved clip with at least 10 frames is reconstructed automatically by `server/recording.ReconJobs`: one clip at a time, in a separate `pocketslam-recon` process, so model memory and GPU work stay out of the live server. Progress: `GET /api/jobs` and the 3D scans tab. `POCKETSLAM_AUTO_RECONSTRUCT=false` turns this off.
- Whole-session recording to `data/recordings/<UTC time>_<session-id>.mcap` is off by default (`POCKETSLAM_RECORD=true` turns it on).
- MCAP topics are fixed. Change them only together with the dashboard and any replay code:

  | Topic | Schema | Content |
  | --- | --- | --- |
  | `/camera/image` | `foxglove.CompressedImage` | JPEG frame |
  | `/camera/calibration` | `foxglove.CameraCalibration` | intrinsics, logged when they change |
  | `/imu/gyro` | JSON `pocketslam.ImuVector` | rad/s |
  | `/imu/accel` | JSON `pocketslam.ImuVector` | m/s², with gravity |
  | `/phone/meta` | JSON `pocketslam.SessionMetadata` | resolution, device, intrinsics source |

- Message `log_time` and `timestamp` are the host-clock time from `ClockMapper`, in nanoseconds.
- JPEG encoding and MCAP writes run on the recording thread, never on the event loop. Its queue is bounded; overflow is dropped and counted (`record_dropped` on the dashboard).
- A session must finish its MCAP file on every exit path: phone disconnect, replacement by a new phone, and SIGTERM.
- One phone at a time. A new offer replaces the active session.
- Ports: phone page **10001** (HTTPS, LAN). Dashboard **8000** on localhost, falling back to **8001** if 8000 is busy. Optional Foxglove WebSocket on **8765** (`POCKETSLAM_LIVE_VIEW=true`).

### 5.6 Offline reconstruction

- `uv run pocketslam-recon <recording.mcap>` turns a recording into depth maps, camera poses and one fused point cloud in `data/reconstructions/<name>/`: `recon.mcap` (Foxglove 3D), `points.ply`, `trajectory.tum`, `depth.npz`, `summary.json`.
- It needs the optional extra: `uv sync --extra recon`. The live server never imports `recon` (import-linter contract).
- Models and licenses. Ids and revisions live in `config.ReconSettings`; change them together.

  | Role | Checkpoint | License |
  | --- | --- | --- |
  | Depth + poses + intrinsics | `depth-anything/DA3-LARGE-1.1` | CC BY-NC 4.0 per upstream README: non-commercial |
  | Metric scale check | `depth-anything/DA3METRIC-LARGE` | Apache 2.0 |
  | Commercial-safe fallback | `depth-anything/DA3-BASE` | Apache 2.0 |

- `depth/da3.py` stubs DA3's export module before importing it. Do not remove the stub: the export helpers pull in `pycolmap`, whose OpenMP runtime aborts the process next to torch's.
- Scale and gravity come from the IMU (`slam/vi_align.py`, VINS-Mono style least squares):
  - Camera–IMU time offset: gyro speed vs visual rotation speed.
  - Phone orientation: the best of four 90° candidates.
  - Scale observability: accel std ≥ 0.2 m/s² and scale uncertainty ≤ 15%.
  - If the IMU scale is not observable, the metric-depth scale is used; `summary.json` says which (`scale_source`).
- Output world: z up (gravity-aligned), metres, camera poses camera-to-world in OpenCV axes.
- Scale sources, in priority order (`recon/reconstruct._choose_scale`): fiducial marker, IMU (when observable), metric depth with the remembered marker correction, metric depth, none.
- Marker scale (`recon/marker.py`): ArUco detection on full-resolution keyframes; per view, a plane is fitted to the depth inside the marker and corner rays are intersected with it; the median over at least 2 views is used and rejected above 20% spread. Marker size is the edge of the **black square**. The triangulated cross-check is reported only with at least 0.2 baseline-to-distance parallax.
- Live calibration (`calib/live.py`, **Calibrate camera** in the dashboard): a worker thread detects the marker in the newest live frame, keeps views that differ by at least 25 px, and re-solves the focal length; it saves when the uncertainty is at most 5%, at least 10 views exist, and both a sideways and a vertical tilt were seen. Guidance goes to the dashboard (WebSocket `calibration` messages) and to the phone (DataChannel). `calib/` holds marker detection and the calibration store so both the live server and `recon/` can use them; the server never imports `recon/`.
- Distance demo (`calib/distance.py`, `calib/demo.py`, **Distance demo** in the dashboard): `DistanceDemo` runs `LiveCalibration` when the phone has no saved focal length (or on "Calibrate again"), then `LiveDistance`. Distance is camera centre to marker centre from perspective-n-point on the four corners with the calibrated focal length. Use `cv2.SOLVEPNP_ITERATIVE` and reject poses that do not reproject within 5 px: OpenCV 5.0's `SOLVEPNP_IPPE_SQUARE` returns wrong poses for this input. Readings are a 5-frame median; the detector uses sub-pixel corner refinement.
- The marker fixes sizes across the image; depth along the view direction is only right if the focal length is right. `pocketslam-calibrate` measures the focal length from tilted marker views and saves it only below 5% uncertainty. The marker's correction for the metric-depth scale is remembered **only** for phones with a measured focal length (`recon/calibration.py`, `data/calibration/scale.json`, keyed by phone model).
- The dashboard's **3D scans** tab (`web/dashboard/scans.js`, plain WebGL2, no libraries) lists `data/reconstructions/` through `GET /api/scans` and loads `points.ply` and `trajectory.tum` through `GET /api/scans/{name}/{file}`. Only those file names are served, and scan names are validated, so the endpoint cannot read outside the folder.
- Record clips for reconstruction with deliberate motion: move the phone 30–50 cm sideways while rotating a little. Pure rotation or a still phone gives no metric scale from the IMU.

## 6. Docker

### 6.1 The macOS limitation

**Docker on macOS has no access to the Apple GPU (Metal/MPS).** A container on a Mac runs inference on CPU only. So:

- On a Mac, run the real-time pipeline **natively**: `uv run python -m pocketslam`.
- Use the Docker image for CI, CPU-only runs, and Linux hosts with NVIDIA GPUs.
- Do not add workarounds that try to reach MPS from a container. None work.

### 6.2 Dockerfile rules [gate]

- Multi-stage build: a `builder` stage installs dependencies with uv, and a `runtime` stage copies only `.venv` and `web/` (the project is installed into `.venv` non-editable).
- The base image is pinned by tag **and** digest (`python:3.12-slim-trixie@sha256:...`). The uv image is pinned the same way. Renovate or Dependabot updates the digests.
- Install dependencies with `uv sync --frozen --no-dev`. Copy `pyproject.toml` and `uv.lock` first and install before you copy source, so the dependency layer stays cached.
- The container runs as a non-root user.
- No secrets, TLS keys, or model weights in any layer. Mount them at runtime.
- `HEALTHCHECK` runs `python -m pocketslam.healthcheck`, which calls `GET /healthz` over HTTP or HTTPS to match the settings.
- `hadolint` passes with no ignores unless a comment explains each one.
- The `runtime` (CPU) target is the default. A `runtime-cuda` target for Linux GPU hosts is planned. It needs CUDA torch wheels, which means separate `cpu` and `cuda` dependency extras declared as conflicting in uv, and a pinned `nvidia/cuda` runtime base. Add it together with the first real depth model.

`docker/Dockerfile` is the reference implementation of these rules. Change the rules and the file together.

### 6.3 WebRTC networking in containers

- aiortc picks random UDP ports for ICE. Publishing a fixed `-p` range does not work reliably.
- On Linux, run with `network_mode: host` in `compose.yaml`.
- If host networking is not possible, run a TURN server (coturn) with a fixed relay port range and point the client at it.
- No STUN servers on the LAN path. aiortc otherwise queries Google's public STUN server on every connection, which leaks the session to a third party and slows connection setup.
- The signaling server binds `0.0.0.0` inside the container so LAN phones can reach it. Outside a container, bind the LAN interface from config.

### 6.4 `docker/Dockerfile.dockerignore`

Must exclude at least: `.git`, `.venv`, `data/`, `**/__pycache__`, `.cache`, `*.pem`, `*.key`, `tests/`.

## 7. Runtime and production rules

- **Config:** all settings come from env vars through `pydantic-settings` in `config.py`, with the `POCKETSLAM_` prefix. No hardcoded IPs, ports, paths, or model names in code.
- **Logging:** use the `logging` module (or `structlog`) with JSON output in production. Never use `print`. Each log line from the pipeline carries the session id.
- **Errors:** never catch a bare `except:` or `except Exception: pass`. One phone disconnecting must not crash the server or other sessions.
- **Shutdown:** handle SIGTERM. Close peer connections, flush the map, and stop workers within 10 s.
- **Security:** the server is LAN-only by default. Any internet exposure needs auth on signaling. Validate and size-limit every DataChannel message.
- **Outputs:** recordings are MCAP files in `data/recordings/` (section 5.5). Maps and trajectories are written to `data/outputs/<session-id>/` in standard formats (PLY for point clouds, TUM format for trajectories).

## 8. Evaluation

- Keep a small set of recorded phone sequences in `data/recordings/` (gitignored, downloaded by script). Each recording has video, timestamps, and metadata, so the pipeline can replay it with no phone.
- Report trajectory error (ATE, RPE) with `evo` against a public benchmark (TUM RGB-D or similar) for any change in `slam/` or `depth/`.
- Report depth error (AbsRel, δ<1.25) for any change in `depth/`.

## 9. Git and CI

- Branch from `main`. Never push directly to `main`.
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `build:`, `ci:`, `perf:`.
- CI (GitHub Actions) runs on every PR: `uv sync --frozen`, all gates from section 3, and the Docker build.
- A separate, manual or nightly CI job runs `gpu` and `slow` tests.
- Do not commit generated files, weights, recordings, certificates, or `.env` files.
