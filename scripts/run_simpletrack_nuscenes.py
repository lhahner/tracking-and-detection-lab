from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_DETECTION_GLOB = "src/detector/*/detections/*.json"
DEFAULT_TRACK_ROOT = Path("src/tracker/SimpleTrack/tracks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SimpleTrack over one or more SimpleTrack-format nuScenes "
            "detection JSON files. By default this scans "
            "src/detector/*/detections/*.json and skips checkpoint JSONL files."
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
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Optional SimpleTrack config YAML. Defaults to the bundled giou.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tracking result files.",
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
        if path.name.endswith(".jsonl"):
            continue
        if path.name.endswith(".tmp"):
            continue
        if not args.include_mini and path.name == "detections.json":
            continue
        filtered_paths.append(path)
    return filtered_paths


def load_detection_payload(detection_path: Path) -> dict[str, Any]:
    with detection_path.open("r", encoding="utf-8") as detection_file:
        payload = json.load(detection_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError(
            f"{detection_path} is not a SimpleTrack detection JSON with a top-level 'frames' list."
        )
    return payload


def default_track_path(detection_path: Path, output_dir: Path) -> Path:
    detector_dir = detection_path.parents[1].name
    stem = detection_path.stem
    if stem.endswith("_nuscenes_simpletrack_val"):
        stem = stem[: -len("_nuscenes_simpletrack_val")]
    elif stem.endswith("_simpletrack_val"):
        stem = stem[: -len("_simpletrack_val")]
    return output_dir / f"{detector_dir}_{stem}_tracks.json"


def run_tracking_file(
    *,
    detection_path: Path,
    output_path: Path,
    config_path: Path | None,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "reason": "output exists",
            "detection_path": detection_path,
            "output_path": output_path,
        }

    from tracker.SimpleTrack import SimpleTrack

    payload = load_detection_payload(detection_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = SimpleTrack(output_path=output_path, config_path=config_path)
    tracking_results = tracker.track(payload)
    return {
        "status": "written",
        "frame_count": len(payload["frames"]),
        "tracking_frame_count": len(tracking_results),
        "detection_path": detection_path,
        "output_path": output_path,
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
            config_path=args.config_path,
            overwrite=args.overwrite,
        )
        if result["status"] == "skipped":
            print(f"Skipped {result['detection_path']} -> {result['output_path']} ({result['reason']})")
        else:
            print(
                f"Tracked {result['frame_count']} frames from {result['detection_path']} "
                f"-> {result['output_path']}"
            )


if __name__ == "__main__":
    main()
