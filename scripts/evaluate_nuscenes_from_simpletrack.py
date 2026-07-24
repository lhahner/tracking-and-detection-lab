from __future__ import annotations

import argparse
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
            "Convert SimpleTrack-format nuScenes detections into the official "
            "results_nusc.json format and optionally run the nuScenes devkit "
            "detection evaluation. The input must be a JSON object with a "
            "'frames' list or a raw list of frame objects. Each frame must "
            "contain 'sample_token' and 'detections'; each detection must "
            "contain 'label', 'score', and 'bbox_3d' in "
            "[x, y, z, yaw, length, width, height, (score)] order."
        )
    )
    parser.add_argument(
        "simpletrack_path",
        type=Path,
        help="SimpleTrack detection JSON file.",
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
        help="nuScenes dataset root. Required to resolve LiDAR-frame SimpleTrack boxes into global-frame results.",
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


def load_simpletrack_frames(simpletrack_path: Path) -> list[dict[str, Any]]:
    with simpletrack_path.open("r", encoding="utf-8") as simpletrack_file:
        payload = json.load(simpletrack_file)

    if isinstance(payload, dict):
        frames = payload.get("frames")
        if frames is None and "frame" in payload and "detections" in payload:
            frames = [payload]
    elif isinstance(payload, list):
        frames = payload
    else:
        frames = None

    if not isinstance(frames, list):
        raise ValueError(
            "SimpleTrack detection JSON must be a frame list, a single frame object, "
            "or an object with a 'frames' list."
        )
    return frames


def parse_simpletrack_box(
    detection: dict[str, Any],
    frame_index: int,
    detection_index: int,
) -> tuple[float, float, float, float, float, float, float, float]:
    bbox = detection.get("bbox_3d")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 7:
        raise ValueError(
            f"Frame {frame_index}, detection {detection_index} must provide "
            "'bbox_3d' with at least 7 values: [x, y, z, yaw, length, width, height, (score)]."
        )

    try:
        x, y, z, yaw, length, width, height = (float(value) for value in bbox[:7])
        score = float(detection.get("score", bbox[7] if len(bbox) > 7 else 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame {frame_index}, detection {detection_index} contains non-numeric "
            "bbox_3d or score values."
        ) from exc

    return x, y, z, yaw, length, width, height, score


def convert_simpletrack_detection_to_nuscenes(
    *,
    frame: dict[str, Any],
    frame_index: int,
    detection: dict[str, Any],
    detection_index: int,
    transform_resolver: NuScenesTransformResolver,
) -> tuple[str, dict[str, Any]]:
    sample_token = frame.get("sample_token")
    if sample_token is None:
        raise ValueError(f"Frame {frame_index} is missing required field 'sample_token'.")
    sample_token = str(sample_token)

    detection_name = detection.get("label")
    if detection_name is None:
        raise ValueError(
            f"Frame {frame_index}, detection {detection_index} is missing required field 'label'."
        )
    detection_name = str(detection_name).lower()
    x, y, z, yaw, length, width, height, detection_score = parse_simpletrack_box(
        detection,
        frame_index,
        detection_index,
    )

    translation, rotation, velocity = transform_resolver.lidar_box_to_global(
        sample_token=sample_token,
        center_lidar=np.asarray([x, y, z], dtype=np.float64),
        yaw_lidar=yaw,
        velocity_lidar=np.asarray([0.0, 0.0], dtype=np.float64),
    )
    detection = {
        "sample_token": sample_token,
        "translation": translation,
        "size": [width, length, height],
        "rotation": rotation,
        "velocity": velocity,
        "detection_name": detection_name,
        "detection_score": detection_score,
        "attribute_name": "",
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


def convert_simpletrack_to_results(
    simpletrack_path: Path,
    meta: dict[str, Any],
    nusc: Any,
    expected_sample_tokens: list[str] | None = None,
) -> dict[str, Any]:
    transform_resolver = NuScenesTransformResolver(nusc)
    expected_token_set = set(expected_sample_tokens or [])
    results: dict[str, list[dict[str, Any]]] = build_seeded_results(expected_sample_tokens or [])
    frames = load_simpletrack_frames(simpletrack_path)

    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(
                f"Frame {frame_index} must be an object with 'sample_token' and 'detections'."
            )
        sample_token = frame.get("sample_token")
        if sample_token is None:
            raise ValueError(f"Frame {frame_index} is missing required field 'sample_token'.")
        sample_token = str(sample_token)
        if expected_token_set and sample_token not in expected_token_set:
            raise ValueError(
                f"Frame {frame_index} references sample_token '{sample_token}' that is not "
                "part of the expected evaluation sample set. Check --eval-set or --sample-list."
            )

        detections = frame.get("detections")
        if detections is None:
            raise ValueError(f"Frame {frame_index} is missing required field 'detections'.")
        if not isinstance(detections, list):
            raise ValueError(f"Frame {frame_index} field 'detections' must be a list.")

        for detection_index, simpletrack_detection in enumerate(detections):
            if not isinstance(simpletrack_detection, dict):
                raise ValueError(
                    f"Frame {frame_index}, detection {detection_index} must be an object."
                )
            _, detection = convert_simpletrack_detection_to_nuscenes(
                frame=frame,
                frame_index=frame_index,
                detection=simpletrack_detection,
                detection_index=detection_index,
                transform_resolver=transform_resolver,
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
    simpletrack_path = args.simpletrack_path.resolve()
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else simpletrack_path.with_name("results_nusc.json")
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else output_json.parent
    )

    if args.dataroot is None:
        raise ValueError(
            "--dataroot is required to convert LiDAR-frame SimpleTrack boxes into global-frame nuScenes results."
        )

    meta = load_meta(args.meta.resolve() if args.meta is not None else None)
    nusc = load_nuscenes(args.version, args.dataroot.resolve())
    expected_sample_tokens = (
        load_sample_tokens_from_file(args.sample_list.resolve())
        if args.sample_list is not None
        else get_eval_sample_tokens(nusc, args.eval_set)
    )
    results_payload = convert_simpletrack_to_results(
        simpletrack_path,
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
