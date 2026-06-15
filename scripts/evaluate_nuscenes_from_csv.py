from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_META = {
    "use_camera": False,
    "use_lidar": True,
    "use_radar": False,
    "use_map": False,
    "use_external": False,
}

FIELD_ALIASES = {
    "sample_token": ("sample_token",),
    "detection_name": ("detection_name", "class_name", "category", "label_name"),
    "detection_score": ("detection_score", "score", "confidence"),
    "x": ("x", "center_x", "translation_x"),
    "y": ("y", "center_y", "translation_y"),
    "z": ("z", "center_z", "translation_z"),
    "length": ("length", "l"),
    "width": ("width", "w"),
    "height": ("height", "h"),
    "yaw": ("yaw", "heading", "rotation_y"),
    "velocity_x": ("velocity_x", "vx"),
    "velocity_y": ("velocity_y", "vy"),
    "attribute_name": ("attribute_name", "attribute"),
}

REQUIRED_FIELDS = (
    "sample_token",
    "detection_name",
    "detection_score",
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "yaw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a flat CSV of nuScenes detections into the official "
            "results_nusc.json format and optionally run the nuScenes devkit "
            "detection evaluation."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV file containing one detection per row.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Destination for the generated results JSON.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="Optional JSON file overriding the default nuScenes meta block.",
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        help="nuScenes dataset root. Required when --run-eval is used.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version to evaluate against, for example v1.0-mini or v1.0-trainval.",
    )
    parser.add_argument(
        "--eval-set",
        default="mini_val",
        help="nuScenes split name for evaluation, for example mini_val or val.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for nuScenes evaluation artifacts. Defaults to the "
            "directory containing the generated JSON."
        ),
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run the official nuScenes detection evaluation after writing JSON.",
    )
    return parser.parse_args()


def load_meta(meta_path: Path | None) -> dict[str, Any]:
    if meta_path is None:
        return dict(DEFAULT_META)
    with meta_path.open("r", encoding="utf-8") as meta_file:
        loaded_meta = json.load(meta_file)
    if not isinstance(loaded_meta, dict):
        raise ValueError("Meta JSON must contain an object at the top level.")
    return loaded_meta


def resolve_field(row: dict[str, str], canonical_name: str) -> str:
    for alias in FIELD_ALIASES[canonical_name]:
        if alias in row and row[alias] != "":
            return row[alias]
    raise KeyError(canonical_name)


def parse_float(row: dict[str, str], canonical_name: str, default: float | None = None) -> float:
    try:
        raw_value = resolve_field(row, canonical_name)
    except KeyError:
        if default is not None:
            return default
        raise
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse field '{canonical_name}' value {raw_value!r} as float."
        ) from exc


def parse_string(row: dict[str, str], canonical_name: str, default: str | None = None) -> str:
    try:
        return resolve_field(row, canonical_name)
    except KeyError:
        if default is not None:
            return default
        raise


def yaw_to_quaternion(yaw: float) -> list[float]:
    half_yaw = yaw / 2.0
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def convert_row_to_nuscenes_detection(row: dict[str, str], row_index: int) -> tuple[str, dict[str, Any]]:
    try:
        sample_token = parse_string(row, "sample_token")
        detection_name = parse_string(row, "detection_name")
        detection_score = parse_float(row, "detection_score")
        x = parse_float(row, "x")
        y = parse_float(row, "y")
        z = parse_float(row, "z")
        length = parse_float(row, "length")
        width = parse_float(row, "width")
        height = parse_float(row, "height")
        yaw = parse_float(row, "yaw")
        velocity_x = parse_float(row, "velocity_x", default=0.0)
        velocity_y = parse_float(row, "velocity_y", default=0.0)
        attribute_name = parse_string(row, "attribute_name", default="")
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Row {row_index} is missing required field '{missing}'. "
            f"Accepted aliases: {FIELD_ALIASES.get(missing, ())}"
        ) from exc

    detection = {
        "sample_token": sample_token,
        "translation": [x, y, z],
        "size": [width, length, height],
        "rotation": yaw_to_quaternion(yaw),
        "velocity": [velocity_x, velocity_y],
        "detection_name": detection_name,
        "detection_score": detection_score,
        "attribute_name": attribute_name,
    }
    return sample_token, detection


def convert_csv_to_results(csv_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} does not contain a header row.")

        normalized_headers = set(reader.fieldnames)
        missing_fields = [
            field_name
            for field_name in REQUIRED_FIELDS
            if not any(alias in normalized_headers for alias in FIELD_ALIASES[field_name])
        ]
        if missing_fields:
            raise ValueError(
                "CSV header is missing required columns for nuScenes export: "
                + ", ".join(missing_fields)
            )

        for row_index, row in enumerate(reader, start=2):
            sample_token, detection = convert_row_to_nuscenes_detection(row, row_index)
            results.setdefault(sample_token, []).append(detection)

    return {"meta": meta, "results": results}


def write_results_json(results_payload: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(results_payload, output_file, indent=2)


def run_nuscenes_evaluation(
    *,
    dataroot: Path,
    version: str,
    eval_set: str,
    result_path: Path,
    output_dir: Path,
) -> None:
    try:
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError as exc:
        raise ImportError(
            "nuScenes evaluation requires nuscenes-devkit to be installed."
        ) from exc

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)
    evaluator = NuScenesEval(
        nusc=nusc,
        config=None,
        result_path=str(result_path),
        eval_set=eval_set,
        output_dir=str(output_dir),
        verbose=True,
    )
    evaluator.main(render_curves=False)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else csv_path.with_name("results_nusc.json")
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else output_json.parent
    )

    meta = load_meta(args.meta.resolve() if args.meta is not None else None)
    results_payload = convert_csv_to_results(csv_path, meta)
    write_results_json(results_payload, output_json)
    detection_count = sum(len(detections) for detections in results_payload["results"].values())
    print(
        f"Wrote {detection_count} detections across "
        f"{len(results_payload['results'])} samples to {output_json}"
    )

    if not args.run_eval:
        return

    if args.dataroot is None:
        raise ValueError("--dataroot is required when --run-eval is used.")

    run_nuscenes_evaluation(
        dataroot=args.dataroot.resolve(),
        version=args.version,
        eval_set=args.eval_set,
        result_path=output_json,
        output_dir=output_dir,
    )
    print(f"nuScenes evaluation finished. Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
