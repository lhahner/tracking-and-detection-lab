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


DEFAULT_TRACK_GLOB = "src/tracker/SimpleTrack/tracks/*.json"
DEFAULT_PROJECT_ROOT = Path(
    "/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate/pad nuScenes tracking result JSON files and optionally run "
            "the official nuScenes tracking devkit evaluation."
        )
    )
    parser.add_argument(
        "track_paths",
        nargs="*",
        type=Path,
        help="Tracking result JSON files. Defaults to --glob when omitted.",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_TRACK_GLOB,
        help=f"Tracking result glob used when no paths are provided. Default: {DEFAULT_TRACK_GLOB}",
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "datasets/nuscenes",
        help="nuScenes dataset root.",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/nuscenes_tracking_eval"),
        help="Directory for nuScenes tracking evaluation artifacts.",
    )
    parser.add_argument(
        "--pad-missing-samples",
        action="store_true",
        help="Insert empty result lists for eval-set sample tokens missing from the result JSON.",
    )
    parser.add_argument(
        "--write-padded",
        action="store_true",
        help="Persist padding changes to the original tracking JSON before evaluation.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run official nuScenes TrackingEval after validation/padding.",
    )
    parser.add_argument(
        "--skip-existing-eval",
        action="store_true",
        help="Skip evaluation if the per-file output directory already contains metrics_summary.json.",
    )
    parser.add_argument(
        "--max-boxes-per-sample",
        type=int,
        default=500,
        help=(
            "Maximum tracks kept per sample before devkit evaluation. "
            "nuScenes requires <= 500. Set to 0 to disable capping."
        ),
    )
    return parser.parse_args()


def discover_track_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.track_paths) if args.track_paths else sorted(Path().glob(args.glob))
    return [path for path in paths if path.suffix == ".json" and path.is_file()]


def load_tracking_payload(track_path: Path) -> dict[str, Any]:
    with track_path.open("r", encoding="utf-8") as track_file:
        payload = json.load(track_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{track_path} must contain a JSON object.")
    meta = payload.get("meta")
    results = payload.get("results")
    if not isinstance(meta, dict):
        raise ValueError(f"{track_path} is missing top-level object field 'meta'.")
    if not isinstance(results, dict):
        raise ValueError(f"{track_path} is missing top-level object field 'results'.")
    for sample_token, sample_tracks in results.items():
        if not isinstance(sample_tracks, list):
            raise ValueError(f"{track_path}: results[{sample_token!r}] must be a list.")
        for track_index, track in enumerate(sample_tracks):
            validate_tracking_box(track_path, sample_token, track_index, track)
    return payload


def validate_tracking_box(
    track_path: Path,
    sample_token: str,
    track_index: int,
    track: Any,
) -> None:
    if not isinstance(track, dict):
        raise ValueError(f"{track_path}: track {track_index} for {sample_token} must be an object.")
    required_fields = {
        "sample_token",
        "translation",
        "size",
        "rotation",
        "velocity",
        "tracking_id",
        "tracking_name",
        "tracking_score",
    }
    missing = sorted(required_fields - set(track))
    if missing:
        raise ValueError(
            f"{track_path}: track {track_index} for {sample_token} is missing {missing}."
        )
    if str(track["sample_token"]) != str(sample_token):
        raise ValueError(
            f"{track_path}: track {track_index} sample_token {track['sample_token']!r} "
            f"does not match results key {sample_token!r}."
        )
    for field_name, expected_len in (("translation", 3), ("size", 3), ("rotation", 4), ("velocity", 2)):
        value = track[field_name]
        if not isinstance(value, list) or len(value) != expected_len:
            raise ValueError(
                f"{track_path}: track {track_index} field {field_name!r} must be a "
                f"list of length {expected_len}."
            )
        for numeric_value in value:
            float(numeric_value)
    float(track["tracking_score"])


def load_eval_sample_tokens(dataroot: Path, version: str, eval_set: str) -> list[str]:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    split_scenes = create_splits_scenes(verbose=False)
    if eval_set not in split_scenes:
        raise ValueError(f"Unknown eval set {eval_set!r}. Available: {sorted(split_scenes)}")

    selected_scene_names = set(split_scenes[eval_set])
    sample_tokens: list[str] = []
    for scene in sorted(nusc.scene, key=lambda record: record["name"]):
        if scene["name"] not in selected_scene_names:
            continue
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample_tokens.append(sample_token)
            sample = nusc.get("sample", sample_token)
            sample_token = sample["next"]
    return sample_tokens


def pad_missing_samples(payload: dict[str, Any], sample_tokens: list[str]) -> int:
    results = payload["results"]
    missing_count = 0
    for sample_token in sample_tokens:
        if sample_token not in results:
            results[sample_token] = []
            missing_count += 1
    return missing_count


def cap_boxes_per_sample(payload: dict[str, Any], max_boxes_per_sample: int) -> tuple[int, int]:
    if max_boxes_per_sample <= 0:
        return 0, 0

    removed_count = 0
    affected_samples = 0
    for sample_token, tracks in payload["results"].items():
        if len(tracks) <= max_boxes_per_sample:
            continue
        tracks.sort(key=lambda track: float(track.get("tracking_score", 0.0)), reverse=True)
        removed_count += len(tracks) - max_boxes_per_sample
        affected_samples += 1
        payload["results"][sample_token] = tracks[:max_boxes_per_sample]
    return removed_count, affected_samples


def write_payload(track_path: Path, payload: dict[str, Any]) -> None:
    track_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = track_path.with_suffix(track_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(track_path)


def run_tracking_eval(
    *,
    track_path: Path,
    dataroot: Path,
    version: str,
    eval_set: str,
    output_dir: Path,
) -> None:
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval

    evaluator = TrackingEval(
        config=config_factory("tracking_nips_2019"),
        result_path=str(track_path),
        eval_set=eval_set,
        output_dir=str(output_dir),
        nusc_dataroot=str(dataroot),
        nusc_version=version,
        verbose=True,
    )
    evaluator.main(render_curves=False)


def output_dir_for_track(base_output_dir: Path, track_path: Path) -> Path:
    return base_output_dir / track_path.stem


def main() -> None:
    args = parse_args()
    track_paths = discover_track_paths(args)
    if not track_paths:
        raise FileNotFoundError("No tracking result JSON files found.")

    sample_tokens = None
    if args.pad_missing_samples or args.run_eval:
        sample_tokens = load_eval_sample_tokens(args.dataroot, args.version, args.eval_set)

    for track_path in track_paths:
        payload = load_tracking_payload(track_path)
        missing_count = 0
        if args.pad_missing_samples:
            missing_count = pad_missing_samples(payload, sample_tokens or [])

        removed_count, capped_samples = cap_boxes_per_sample(
            payload,
            args.max_boxes_per_sample,
        )

        result_count = len(payload["results"])
        box_count = sum(len(tracks) for tracks in payload["results"].values())
        print(
            f"Validated {track_path}: {box_count} tracks across {result_count} samples "
            f"({missing_count} padded missing samples, {removed_count} tracks removed "
            f"from {capped_samples} over-limit samples)."
        )

        if not args.run_eval:
            if args.write_padded and (missing_count or removed_count):
                write_payload(track_path, payload)
            continue
        if box_count == 0:
            raise ValueError(
                f"{track_path} contains zero tracking boxes. The nuScenes tracking "
                "devkit cannot evaluate an all-empty prediction file; rerun SimpleTrack "
                "or choose a tracking result file with non-empty per-sample tracks."
            )

        per_file_output_dir = output_dir_for_track(args.output_dir, track_path)
        metrics_path = per_file_output_dir / "metrics_summary.json"
        if args.skip_existing_eval and metrics_path.exists():
            print(f"Skipped existing evaluation for {track_path}: {metrics_path}")
            continue
        per_file_output_dir.mkdir(parents=True, exist_ok=True)

        eval_track_path = track_path
        if missing_count or removed_count:
            if args.write_padded:
                write_payload(track_path, payload)
            else:
                eval_track_path = per_file_output_dir / f"{track_path.stem}_eval_input.json"
                write_payload(eval_track_path, payload)

        run_tracking_eval(
            track_path=eval_track_path,
            dataroot=args.dataroot,
            version=args.version,
            eval_set=args.eval_set,
            output_dir=per_file_output_dir,
        )
        print(f"nuScenes tracking evaluation finished for {track_path}: {per_file_output_dir}")


if __name__ == "__main__":
    main()
