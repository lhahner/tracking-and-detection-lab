from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_DETECTION_GLOB = "src/detector/*/detections/*.json"
DEFAULT_TRACK_ROOT = Path("src/tracker/SimpleTrack/tracks")
TRACKING_META = {
    "use_camera": False,
    "use_lidar": True,
    "use_radar": False,
    "use_map": False,
    "use_external": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SimpleTrack over SimpleTrack-format nuScenes detection JSON files. "
            "Tracks are checkpointed frame-by-frame and the final nuScenes tracking "
            "JSON is updated during the run so progress is visible on disk."
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
        help="Overwrite existing tracking result and checkpoint files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume writing from an existing JSONL checkpoint. Earlier frames are "
            "replayed to restore tracker state, but already checkpointed samples are not rewritten."
        ),
    )
    parser.add_argument(
        "--include-mini",
        action="store_true",
        help="Also process generic mini-test files such as detections.json.",
    )
    parser.add_argument(
        "--write-interval",
        type=int,
        default=10,
        help="Refresh the final tracking JSON every N newly written frames. Default: 10.",
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
    return output_dir / f"{detector_dir}_{stem}_tracks.json"


def initialize_tracking_json(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"meta": TRACKING_META, "results": {}}, indent=2),
        encoding="utf-8",
    )


def load_completed_tokens(checkpoint_path: Path) -> set[str]:
    completed_tokens: set[str] = set()
    if not checkpoint_path.exists():
        return completed_tokens
    with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
        for line_number, line in enumerate(checkpoint_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} contains invalid JSON on line {line_number}."
                ) from exc
            sample_token = record.get("sample_token")
            if sample_token is not None:
                completed_tokens.add(str(sample_token))
    return completed_tokens


def append_checkpoint_record(
    checkpoint_file,
    sample_token: str,
    tracks: list[dict[str, Any]],
) -> None:
    checkpoint_file.write(json.dumps({"sample_token": sample_token, "tracks": tracks}) + "\n")
    checkpoint_file.flush()


def assemble_tracking_json(checkpoint_path: Path, output_path: Path) -> int:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    sample_count = 0
    with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            output_file.write('{\n  "meta": ')
            json.dump(TRACKING_META, output_file, indent=2)
            output_file.write(',\n  "results": {')
            first_sample = True
            for line in checkpoint_file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_token = str(record["sample_token"])
                tracks = record.get("tracks", [])
                if not first_sample:
                    output_file.write(",")
                output_file.write("\n    ")
                json.dump(sample_token, output_file)
                output_file.write(": ")
                json.dump(tracks, output_file)
                first_sample = False
                sample_count += 1
            output_file.write("\n  }\n}\n")
    temporary_path.replace(output_path)
    return sample_count


def iter_simpletrack_frames(detection_path: Path) -> Iterator[dict[str, Any]]:
    with detection_path.open("r", encoding="utf-8") as detection_file:
        found_frames_key = False
        in_string = False
        escaped = False
        recent = ""

        while True:
            char = detection_file.read(1)
            if not char:
                raise ValueError(f"{detection_path} does not contain a top-level 'frames' array.")

            recent = (recent + char)[-16:]
            if not found_frames_key and '"frames"' in recent:
                found_frames_key = True

            if found_frames_key and char == "[":
                break

        while True:
            char = detection_file.read(1)
            if not char:
                return
            if char.isspace() or char == ",":
                continue
            if char == "]":
                return
            if char != "{":
                raise ValueError(f"Expected frame object in {detection_path}, got {char!r}.")

            buffer = [char]
            depth = 1
            in_string = False
            escaped = False
            while depth > 0:
                char = detection_file.read(1)
                if not char:
                    raise ValueError(f"Unexpected EOF while reading frame object in {detection_path}.")
                buffer.append(char)

                if escaped:
                    escaped = False
                    continue
                if char == "\\" and in_string:
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

            frame = json.loads("".join(buffer))
            if not isinstance(frame, dict):
                raise ValueError(f"Frame entry in {detection_path} is not a JSON object.")
            yield frame


def extract_single_sample_result(temp_result_path: Path) -> tuple[str, list[dict[str, Any]]]:
    with temp_result_path.open("r", encoding="utf-8") as temp_file:
        payload = json.load(temp_file)
    results = payload.get("results")
    if not isinstance(results, dict) or len(results) != 1:
        raise ValueError(f"Expected exactly one sample result in {temp_result_path}.")
    sample_token, tracks = next(iter(results.items()))
    if not isinstance(tracks, list):
        raise ValueError(f"Expected tracks for sample {sample_token} to be a list.")
    return str(sample_token), tracks


def run_tracking_file(
    *,
    detection_path: Path,
    output_path: Path,
    config_path: Path | None,
    overwrite: bool,
    resume: bool,
    write_interval: int,
) -> dict[str, Any]:
    from tracker.SimpleTrack import SimpleTrack

    checkpoint_path = output_path.with_suffix(output_path.suffix + ".jsonl")
    temp_result_path = output_path.with_suffix(output_path.suffix + ".frame.tmp")

    if output_path.exists() and checkpoint_path.exists() and not overwrite and not resume:
        return {
            "status": "skipped",
            "reason": "output and checkpoint exist",
            "detection_path": detection_path,
            "output_path": output_path,
        }

    if overwrite:
        output_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
        temp_result_path.unlink(missing_ok=True)

    completed_tokens = load_completed_tokens(checkpoint_path) if resume else set()
    initialize_tracking_json(output_path)
    if completed_tokens:
        assemble_tracking_json(checkpoint_path, output_path)

    tracker = SimpleTrack(output_path=temp_result_path, config_path=config_path)
    processed_frames = 0
    written_frames = 0
    mode = "a" if resume and checkpoint_path.exists() else "w"
    with checkpoint_path.open(mode, encoding="utf-8") as checkpoint_file:
        for frame in iter_simpletrack_frames(detection_path):
            processed_frames += 1
            tracker.track({"frames": [frame]})
            sample_token, tracks = extract_single_sample_result(temp_result_path)

            if sample_token in completed_tokens:
                continue

            append_checkpoint_record(checkpoint_file, sample_token, tracks)
            written_frames += 1
            if written_frames % max(write_interval, 1) == 0:
                assembled_count = assemble_tracking_json(checkpoint_path, output_path)
                print(
                    f"updated {output_path} with {assembled_count} tracked samples "
                    f"after processing {processed_frames} detection frames",
                    flush=True,
                )

    assembled_count = assemble_tracking_json(checkpoint_path, output_path)
    temp_result_path.unlink(missing_ok=True)
    return {
        "status": "written",
        "processed_frames": processed_frames,
        "written_frames": written_frames,
        "assembled_samples": assembled_count,
        "detection_path": detection_path,
        "output_path": output_path,
        "checkpoint_path": checkpoint_path,
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
            resume=args.resume,
            write_interval=args.write_interval,
        )
        if result["status"] == "skipped":
            print(f"Skipped {result['detection_path']} -> {result['output_path']} ({result['reason']})")
        else:
            print(
                f"Tracked {result['processed_frames']} detection frames from {result['detection_path']} "
                f"-> {result['output_path']} ({result['assembled_samples']} samples assembled; "
                f"checkpoint {result['checkpoint_path']})"
            )


if __name__ == "__main__":
    main()
