from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


class NuScenesTransformResolver:
    def __init__(self, nusc: Any):
        self.nusc = nusc
        self._lidar_global_transforms: dict[str, np.ndarray] = {}
        self._lidar_global_rotations: dict[str, np.ndarray] = {}

    def lidar_box_to_global(
        self,
        sample_token: str,
        center_lidar: np.ndarray,
        yaw_lidar: float,
        velocity_lidar: np.ndarray,
    ) -> tuple[list[float], list[float], list[float]]:
        transform = self._get_lidar_global_transform(sample_token)
        sensor_rotation = self._get_lidar_global_rotation(sample_token)

        center_global = transform_points(center_lidar.reshape(1, 3), transform)[0]
        local_yaw_quaternion = yaw_to_quaternion(yaw_lidar)
        global_rotation = quaternion_multiply(sensor_rotation, local_yaw_quaternion)
        global_rotation = normalize_quaternion(global_rotation)

        velocity_xyz = np.array([velocity_lidar[0], velocity_lidar[1], 0.0], dtype=np.float64)
        velocity_global_xyz = quaternion_rotation_matrix(sensor_rotation) @ velocity_xyz
        velocity_global = velocity_global_xyz[:2].tolist()

        return center_global.tolist(), global_rotation.tolist(), velocity_global

    def _get_lidar_global_transform(self, sample_token: str) -> np.ndarray:
        if sample_token not in self._lidar_global_transforms:
            self._populate_sample_cache(sample_token)
        return self._lidar_global_transforms[sample_token]

    def _get_lidar_global_rotation(self, sample_token: str) -> np.ndarray:
        if sample_token not in self._lidar_global_rotations:
            self._populate_sample_cache(sample_token)
        return self._lidar_global_rotations[sample_token]

    def _populate_sample_cache(self, sample_token: str) -> None:
        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_record = self.nusc.get("sample_data", lidar_token)
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", lidar_record["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", lidar_record["ego_pose_token"])
        sensor_to_ego = transform_matrix(
            calibrated_sensor["translation"], calibrated_sensor["rotation"]
        )
        ego_to_global = transform_matrix(ego_pose["translation"], ego_pose["rotation"])
        sensor_to_global = ego_to_global @ sensor_to_ego
        sensor_rotation = quaternion_multiply(
            np.asarray(ego_pose["rotation"], dtype=np.float64),
            np.asarray(calibrated_sensor["rotation"], dtype=np.float64),
        )
        self._lidar_global_transforms[sample_token] = sensor_to_global
        self._lidar_global_rotations[sample_token] = normalize_quaternion(sensor_rotation)


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
        help="nuScenes dataset root. Required to resolve LiDAR-frame CSV boxes into global-frame results.",
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
        "--sample-list",
        type=Path,
        help=(
            "Optional text file containing one nuScenes sample token per line. "
            "When provided, the exporter seeds empty result lists for every token in the file."
        ),
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run the official nuScenes detection evaluation after writing JSON.",
    )
    parser.add_argument(
        "--eval-config",
        default="detection_cvpr_2019",
        help=(
            "nuScenes detection evaluation config name passed to config_factory, "
            "for example detection_cvpr_2019."
        ),
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


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    half_yaw = yaw / 2.0
    return np.asarray([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float64)


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Quaternion must have a non-zero norm")
    return quaternion / norm


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(lhs, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(rhs, dtype=np.float64)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_matrix(translation: list[float], quaternion: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_rotation_matrix(np.asarray(quaternion, dtype=np.float64))
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def convert_row_to_nuscenes_detection(
    row: dict[str, str],
    row_index: int,
    transform_resolver: NuScenesTransformResolver,
) -> tuple[str, dict[str, Any]]:
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

    translation, rotation, velocity = transform_resolver.lidar_box_to_global(
        sample_token=sample_token,
        center_lidar=np.asarray([x, y, z], dtype=np.float64),
        yaw_lidar=yaw,
        velocity_lidar=np.asarray([velocity_x, velocity_y], dtype=np.float64),
    )
    detection = {
        "sample_token": sample_token,
        "translation": translation,
        "size": [width, length, height],
        "rotation": rotation,
        "velocity": velocity,
        "detection_name": detection_name,
        "detection_score": detection_score,
        "attribute_name": attribute_name,
    }
    return sample_token, detection


def load_sample_tokens_from_file(sample_list_path: Path) -> list[str]:
    return [
        line.strip()
        for line in sample_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_eval_sample_tokens(nusc: Any, eval_set: str) -> list[str]:
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise ImportError(
            "nuScenes split seeding requires nuscenes-devkit to be installed."
        ) from exc

    split_scenes = create_splits_scenes(verbose=False)
    if eval_set not in split_scenes:
        raise ValueError(
            f"Unknown nuScenes eval split '{eval_set}'. Available splits: {sorted(split_scenes)}"
        )

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


def build_seeded_results(expected_sample_tokens: list[str]) -> dict[str, list[dict[str, Any]]]:
    return {sample_token: [] for sample_token in expected_sample_tokens}


def convert_csv_to_results(
    csv_path: Path,
    meta: dict[str, Any],
    nusc: Any,
    expected_sample_tokens: list[str] | None = None,
) -> dict[str, Any]:
    transform_resolver = NuScenesTransformResolver(nusc)
    results: dict[str, list[dict[str, Any]]] = build_seeded_results(expected_sample_tokens or [])
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
            sample_token, detection = convert_row_to_nuscenes_detection(
                row,
                row_index,
                transform_resolver,
            )
            results.setdefault(sample_token, []).append(detection)

    return {"meta": meta, "results": results}


def write_results_json(results_payload: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(results_payload, output_file, indent=2)


def load_nuscenes(version: str, dataroot: Path) -> Any:
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise ImportError(
            "nuScenes conversion requires nuscenes-devkit to be installed."
        ) from exc
    return NuScenes(version=version, dataroot=str(dataroot), verbose=True)


def load_detection_eval_config(config_name: str) -> Any:
    try:
        from nuscenes.eval.common.config import config_factory
    except ImportError as exc:
        raise ImportError(
            "nuScenes evaluation requires nuscenes-devkit to be installed."
        ) from exc
    return config_factory(config_name)


def run_nuscenes_evaluation(
    *,
    nusc: Any,
    eval_set: str,
    result_path: Path,
    output_dir: Path,
    eval_config_name: str,
) -> None:
    try:
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError as exc:
        raise ImportError(
            "nuScenes evaluation requires nuscenes-devkit to be installed."
        ) from exc

    evaluator = NuScenesEval(
        nusc=nusc,
        config=load_detection_eval_config(eval_config_name),
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

    if args.dataroot is None:
        raise ValueError(
            "--dataroot is required to convert LiDAR-frame CSV boxes into global-frame nuScenes results."
        )

    meta = load_meta(args.meta.resolve() if args.meta is not None else None)
    nusc = load_nuscenes(args.version, args.dataroot.resolve())
    expected_sample_tokens = (
        load_sample_tokens_from_file(args.sample_list.resolve())
        if args.sample_list is not None
        else get_eval_sample_tokens(nusc, args.eval_set)
    )
    results_payload = convert_csv_to_results(
        csv_path,
        meta,
        nusc,
        expected_sample_tokens=expected_sample_tokens,
    )
    write_results_json(results_payload, output_json)
    detection_count = sum(len(detections) for detections in results_payload["results"].values())
    print(
        f"Wrote {detection_count} detections across "
        f"{len(results_payload['results'])} samples to {output_json}"
    )

    if not args.run_eval:
        return

    run_nuscenes_evaluation(
        nusc=nusc,
        eval_set=args.eval_set,
        result_path=output_json,
        output_dir=output_dir,
        eval_config_name=args.eval_config,
    )
    print(f"nuScenes evaluation finished. Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
