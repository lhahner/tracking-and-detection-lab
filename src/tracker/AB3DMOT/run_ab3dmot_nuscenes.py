from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DETECTION_GLOB = "src/detector/*/detections/*.json"
DEFAULT_TRACK_ROOT = Path("src/tracker/AB3DMOT/tracks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AB3DMOT over SimpleTrack-format nuScenes detection JSON files and write "
            "nuScenes tracking JSON results plus intermediate formatted detection txt files."
        )
    )
    parser.add_argument(
        "detection_paths",
        nargs="*",
        type=Path,
        help="Explicit SimpleTrack-format detection JSON files to process.",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_DETECTION_GLOB,
        help=f"Detection glob used when no paths are provided. Default: {DEFAULT_DETECTION_GLOB}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRACK_ROOT,
        help=f"Directory for tracking result JSON files. Default: {DEFAULT_TRACK_ROOT}",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--detector-name",
        default="centerpoint",
        help="AB3DMOT parameter profile to use. Unsupported names fall back to centerpoint.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Optional AB3DMOT YAML config. Defaults to the bundled nuScenes config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tracking result JSON files.",
    )
    parser.add_argument(
        "--include-mini",
        action="store_true",
        help="Also process generic mini-test files such as detections.json.",
    )
    return parser.parse_args()


def discover_detection_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.detection_paths)
    if not paths:
        paths = sorted(Path().glob(args.glob))

    filtered_paths: list[Path] = []
    for path in paths:
        if path.suffix != ".json":
            continue
        if path.name.endswith(".jsonl") or path.name.endswith(".tmp"):
            continue
        if not args.include_mini and path.name == "detections.json":
            continue
        filtered_paths.append(path)
    return filtered_paths


def default_track_path(detection_path: Path, output_dir: Path) -> Path:
    detector_dir = detection_path.parents[1].name
    stem = detection_path.stem
    for suffix in ("_nuscenes_simpletrack_val", "_simpletrack_val"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return output_dir / f"{detector_dir}_{stem}_ab3dmot_tracks.json"


def run_tracking_file(
    *,
    detection_path: Path,
    output_path: Path,
    split: str,
    detector_name: str,
    config_path: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    from tracker.AB3DMOT import AB3DMOT

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "reason": "output exists",
            "detection_path": detection_path,
            "output_path": output_path,
        }

    if overwrite:
        output_path.unlink(missing_ok=True)

    tracker = AB3DMOT(
        output_path=output_path,
        split=split,
        detector_name=detector_name,
        config_path=config_path,
    )
    tracking_results = tracker.track(detection_path)
    return {
        "status": "written",
        "detection_path": detection_path,
        "output_path": output_path,
        "tracked_frames": len(tracking_results),
    }


def main() -> None:
    args = parse_args()
    detection_paths = discover_detection_paths(args)
    if not detection_paths:
        raise FileNotFoundError("No SimpleTrack detection JSON files found.")

    for detection_path in detection_paths:
        output_path = default_track_path(detection_path, args.output_dir)
        result = run_tracking_file(
            detection_path=detection_path,
            output_path=output_path,
            split=args.split,
            detector_name=args.detector_name,
            config_path=args.config_path,
            overwrite=args.overwrite,
        )
        if result["status"] == "skipped":
            print(f"Skipped {result['detection_path']} -> {result['output_path']} ({result['reason']})")
        else:
            print(
                f"Tracked {result['tracked_frames']} frames from {result['detection_path']} "
                f"-> {result['output_path']}"
            )


if __name__ == "__main__":
    main()
