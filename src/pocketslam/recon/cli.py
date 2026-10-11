"""``pocketslam-recon``: reconstruct a recording into a 3D point cloud.

Example::

    uv sync --extra recon
    uv run pocketslam-recon data/recordings/20261010T083852Z_214448f941c7.mcap
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from pocketslam.config import ReconSettings
from pocketslam.core.jsonsafe import dumps
from pocketslam.log import configure_logging

logger = logging.getLogger("pocketslam.recon")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments; unset options fall back to ``ReconSettings``."""
    p = argparse.ArgumentParser(prog="pocketslam-recon", description=__doc__.splitlines()[0])
    p.add_argument("recording", type=Path, help="MCAP file from data/recordings/")
    p.add_argument(
        "--out", type=Path, help="output directory (default: data/reconstructions/<name>)"
    )
    p.add_argument("--max-frames", type=int, help="most keyframes sent to the model")
    p.add_argument("--start", type=float, default=0.0, help="skip this many seconds at the start")
    p.add_argument("--duration", type=float, help="use only this many seconds")
    p.add_argument("--no-metric", action="store_true", help="skip the metric-depth scale check")
    p.add_argument(
        "--marker-size",
        type=float,
        help="edge of the ArUco marker's black square in metres, e.g. 0.16; sets the scale",
    )
    p.add_argument("--marker-id", type=int, help="only use the marker with this id")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the reconstruction; returns a process exit code."""
    args = parse_args(argv)
    configure_logging("INFO", "text")
    settings = ReconSettings()
    updates: dict[str, object] = {}
    if args.max_frames:
        updates["max_frames"] = args.max_frames
    if args.no_metric:
        updates["use_metric_check"] = False
    if args.marker_size:
        updates["marker_size_m"] = args.marker_size
    if args.marker_id is not None:
        updates["marker_id"] = args.marker_id
    settings = settings.model_copy(update=updates)

    # Heavy imports after argument parsing so --help stays fast.
    from pocketslam.calib.marker import make_detector  # noqa: PLC0415
    from pocketslam.calib.store import ScaleCalibration  # noqa: PLC0415
    from pocketslam.core.device import select_device  # noqa: PLC0415
    from pocketslam.depth.da3 import DA3Metric, DA3MultiView, load_model  # noqa: PLC0415
    from pocketslam.recon.export import export  # noqa: PLC0415
    from pocketslam.recon.reader import read_recording  # noqa: PLC0415
    from pocketslam.recon.reconstruct import reconstruct  # noqa: PLC0415

    recording = read_recording(args.recording).clip(args.start, args.duration)
    logger.info(
        "recording loaded",
        extra={"frames": len(recording), "seconds": round(recording.duration_s, 2)},
    )
    device = select_device(settings.device)
    multiview = DA3MultiView(
        load_model(settings.model, settings.model_revision, device), settings.process_res
    )
    metric = (
        DA3Metric(
            load_model(settings.metric_model, settings.metric_revision, device),
            settings.process_res,
        )
        if settings.use_metric_check
        else None
    )
    detect = (
        make_detector(settings.marker_dictionary, settings.marker_id)
        if settings.marker_size_m is not None
        else None
    )
    result = reconstruct(
        recording,
        multiview,
        metric,
        settings,
        detect=detect,
        calibration=ScaleCalibration(settings.calibration_file),
    )
    out_dir = args.out or settings.output_dir / args.recording.stem
    paths = export(result, out_dir)
    sys.stdout.write(dumps(result.summary, indent=2) + "\n")
    sys.stdout.write(f"\nOpen in Foxglove: {paths['mcap']}\nPoint cloud: {paths['ply']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def marker_main(argv: list[str] | None = None) -> int:
    """``pocketslam-marker``: write a clean marker image with its white border."""
    from PIL import Image  # noqa: PLC0415

    from pocketslam.calib.marker import render_marker  # noqa: PLC0415

    settings = ReconSettings()
    p = argparse.ArgumentParser(
        prog="pocketslam-marker",
        description="Write an ArUco marker image with the white border detection needs.",
    )
    p.add_argument("--id", type=int, default=settings.marker_id or 18, help="marker id")
    p.add_argument("--dictionary", default=settings.marker_dictionary)
    p.add_argument("--side-px", type=int, default=1000, help="black square edge in pixels")
    p.add_argument("--out", type=Path, default=Path("marker.png"))
    args = p.parse_args(argv)
    Image.fromarray(render_marker(args.dictionary, args.id, args.side_px)).save(args.out)
    sys.stdout.write(
        f"Wrote {args.out} ({args.dictionary}, id {args.id}).\n"
        "Show it on a screen or print it, then measure the BLACK square edge with a ruler\n"
        "and set POCKETSLAM_RECON_MARKER_SIZE_M to that length in metres.\n"
    )
    return 0


def calibrate_main(argv: list[str] | None = None) -> int:
    """``pocketslam-calibrate``: measure the phone camera's focal length from a marker clip."""
    from pocketslam.calib.marker import estimate_focal, make_detector  # noqa: PLC0415
    from pocketslam.calib.store import ScaleCalibration, device_label  # noqa: PLC0415
    from pocketslam.recon.reader import read_recording  # noqa: PLC0415

    settings = ReconSettings()
    p = argparse.ArgumentParser(
        prog="pocketslam-calibrate",
        description=(
            "Measure the phone camera's focal length from a clip of the marker. Record the "
            "marker large in the image and from steep angles (30-45 degrees), about 10 seconds."
        ),
    )
    p.add_argument("recording", type=Path, help="MCAP clip that shows the marker")
    p.add_argument("--marker-size", type=float, default=settings.marker_size_m)
    p.add_argument("--marker-id", type=int, default=settings.marker_id)
    args = p.parse_args(argv)
    configure_logging("INFO", "text")
    if args.marker_size is None:
        p.error("set --marker-size (metres) or POCKETSLAM_RECON_MARKER_SIZE_M")

    recording = read_recording(args.recording)
    detect = make_detector(settings.marker_dictionary, args.marker_id)
    corner_sets = []
    size = (0, 0)
    for i in range(len(recording)):
        image = recording.decode(i)
        size = (image.shape[1], image.shape[0])
        found = detect(image)
        if found is not None:
            corner_sets.append(found[1])
    estimate = estimate_focal(corner_sets, size, args.marker_size)
    if estimate is None:
        sys.stdout.write(
            f"Marker found in {len(corner_sets)} of {len(recording)} frames: too few.\n"
        )
        return 1
    fov = 2 * math.degrees(math.atan(max(size) / 2 / estimate.focal_px))
    sys.stdout.write(
        f"Marker found in {len(corner_sets)} of {len(recording)} frames.\n"
        f"Focal length: {estimate.focal_px:.0f} px at {size[0]}x{size[1]} "
        f"(field of view {fov:.1f} degrees on the long side), "
        f"uncertainty {estimate.rel_std:.1%}, fit error {estimate.rms_px:.2f} px.\n"
    )
    if not estimate.reliable:
        sys.stdout.write(f"Not saved: {estimate.reason}.\n")
        return 1
    device = device_label(recording.meta)
    ScaleCalibration(settings.calibration_file).add_focal(
        device, estimate.focal_px / max(size), estimate.rel_std, args.recording.stem
    )
    sys.stdout.write(f"Saved for {device!r} in {settings.calibration_file}.\n")
    return 0
