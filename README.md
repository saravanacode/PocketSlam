# PocketSlam

Device-agnostic monocular depth and SLAM. Any phone streams its camera and motion sensors over WebRTC to a laptop, which records them and runs depth estimation and SLAM.

The phone needs only a browser. The laptop picks CUDA, MPS, or CPU automatically.

> Status: step one. Phone capture, MCAP recording and the live dashboard work. Depth (`ConstantDepthEstimator`) and SLAM (`StaticSlam`) are placeholders.

## Record from your phone

1. Make a certificate for the laptop's LAN address (phones only allow camera and sensors over HTTPS):

   ```bash
   scripts/gen_dev_cert.sh
   cp config/env.example config/.env
   ```

   Without mkcert this makes a self-signed certificate. That is fine: Chrome warns once, then works.

2. Start the laptop server:

   ```bash
   uv run python -m pocketslam
   ```

   It prints two addresses: the phone page and the dashboard.

3. Open the dashboard on the laptop: <http://127.0.0.1:8000> (or 8001 if 8000 is taken). It shows the phone address to open.

4. On the phone, open `https://<laptop-ip>:10001` in Chrome. Phone and laptop must be on the same Wi-Fi. If Chrome warns about the certificate, tap **Advanced**, then **Proceed**. Tap **Start streaming** and allow the camera.

The dashboard shows the live camera and the gyroscope and accelerometer traces.

### Record a clip and get a 3D scan

1. While streaming, press **Record** on the phone (or the **Record** button in the dashboard's top strip).
2. Move the phone slowly: 30–50 cm sideways while turning it a little. 5–10 seconds is enough.
3. Press **Stop recording**. The clip is saved to `data/recordings/clip_<time>.mcap`.
4. The laptop reconstructs it automatically (about a minute). Watch the **3D scans** tab: the job shows as "Reconstructing", and the scan opens when it is ready.

Automatic reconstruction needs Depth Anything 3 installed once: `uv sync --extra recon`. After that, `uv run` keeps it; a plain `uv sync` removes it, so pass `--extra recon` if you sync again. To turn reconstruction off: `POCKETSLAM_AUTO_RECONSTRUCT=false`. To also save every whole session (large files): `POCKETSLAM_RECORD=true`.

At 1280×720 the recording takes about 2 MB/s (JPEG frames). Lower it with `POCKETSLAM_JPEG_QUALITY=70`.

### Look at a recording

Open the `.mcap` file in [Foxglove](https://foxglove.dev) (desktop app or app.foxglove.dev). Topics:

| Topic | Content |
| --- | --- |
| `/camera/image` | JPEG frames |
| `/camera/calibration` | camera intrinsics |
| `/imu/gyro` | gyroscope, rad/s |
| `/imu/accel` | accelerometer with gravity, m/s² |
| `/phone/meta` | resolution and device |

To watch live in Foxglove too, set `POCKETSLAM_LIVE_VIEW=true` and connect Foxglove to `ws://127.0.0.1:8765`.

## Turn a recording into 3D

Offline reconstruction runs Depth Anything 3 on a clip's frames for depth and camera poses, then uses the phone's accelerometer and gyroscope for metric scale and a level (z-up) world.

```bash
uv sync --extra recon                         # once: installs Depth Anything 3
uv run pocketslam-recon data/recordings/<file>.mcap --duration 5
```

- `--start 15 --duration 5` picks a window from a longer recording.
- The first run downloads about 2.8 GB of model weights.
- For a good result, move the phone 30–50 cm sideways while turning it a little. Don't just spin in place.

Outputs land in `data/reconstructions/<file>/`:

| File | What it is |
| --- | --- |
| `recon.mcap` | open in Foxglove: point cloud, camera path, depth images (3D panel, fixed frame `world`) |
| `points.ply` | the fused point cloud for any 3D tool (MeshLab, Blender, CloudCompare) |
| `trajectory.tum` | camera path, TUM format |
| `depth.npz` | per-frame depth in metres, confidences, intrinsics, poses |
| `summary.json` | scale and where it came from (IMU or metric depth), time offset, timings |

To explore a scan, open the dashboard (<http://127.0.0.1:8000>) and switch to the **3D scans** tab:
- drag to orbit, right-drag or Shift-drag to pan, scroll to zoom
- W A S D Q E to move, double-click a point to fly to it
- **Measure**: click two points to get the distance (in metres when the scan has a metric scale)

The tab lists every folder in `data/reconstructions/`; press **Refresh** after a new `pocketslam-recon` run.

### Real-world scale from a marker

A printed or on-screen ArUco marker of known size makes the scan's scale exact.

1. Show the marker with a white border around it. Detection fails if the black square sits on a dark background (for example an image viewer in dark mode). The easy way: on the second screen open `https://<laptop-ip>:10001/marker`. It shows the marker on a white page with a size control; hold a ruler to the screen and set the **black square's edge** to your marker size. To print instead: `uv run pocketslam-marker --id 18 --out marker.png`.
2. Put the size in `config/.env` (metres): `POCKETSLAM_RECON_MARKER_SIZE_M=0.16` and `POCKETSLAM_RECON_MARKER_ID=18`.
3. **Calibrate the phone camera once, live.** While streaming, press **Calibrate camera** in the dashboard's top strip and point the phone at the marker. Follow the hint (shown on the dashboard and on the phone): get the marker to fill about a third of the picture, then view it from the left, right, above and below at steep angles. It saves by itself when the focal length is known to within 5%, usually in under 20 seconds. This is per phone, not per scan; redo it only if you change phone or stream resolution.
   The same can be done from a saved clip: `uv run pocketslam-calibrate data/recordings/<clip>.mcap`.
4. Record scans with the marker somewhere in view for a few seconds. `summary.json` then says `scale_source: "marker"`, and the 3D scans tab outlines the marker in cyan; **Measure** across it should read its size.

### Distance demo

Press **Distance demo** in the dashboard's top strip while the phone streams.

- If the phone is not calibrated yet, it runs the calibration first and then switches by itself.
- It then shows the **distance from the phone's camera to the marker's centre**, live, on the dashboard (with a 30-second trace) and large on the phone's screen. Walk back and watch it change.
- Keep the marker in the picture. If it leaves, the reading greys out and keeps the last value.
- Range with a 16 cm marker at 720×1280: about 0.25 m to about 7 m. Expect well under 1% error up to 2 m and a few percent beyond 3 m, where the marker is only a few dozen pixels wide; the readout shows its own error estimate.
- **Calibrate again** re-measures the focal length before measuring.

It does not track distance once the marker is out of view, and it does not save the readings into recordings.

Scale priority: marker, then IMU, then the depth model. Once the focal length is measured, each marker scan also teaches a correction for the depth model's scale (`data/calibration/scale.json`), which is applied to scans without a marker (`scale_source: "metric-depth (calibrated)"`).

Without step 3 the marker scale still applies, but it relies on the model's guess of the focal length, so distances can be off by tens of percent; the 3D tab says so.

The default depth-and-pose model (`DA3-LARGE-1.1`) is non-commercial. For commercial use set `POCKETSLAM_RECON_MODEL=depth-anything/DA3-BASE` with its revision (see `src/pocketslam/config.py`).

## Docker (Linux hosts)

Run natively on a Mac: Docker on macOS cannot use the Apple GPU.

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
