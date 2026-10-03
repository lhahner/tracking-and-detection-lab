"""Convert nuScenes detection results into chronological SimpleTrack frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("metadata_template", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def quaternion_rotation_matrix(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Detection quaternion must have non-zero norm.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def convert_detection(detection: dict, global_to_lidar: np.ndarray) -> dict:
    translation_global = np.asarray([*detection["translation"], 1.0], dtype=np.float64)
    translation_lidar = global_to_lidar @ translation_global

    rotation_global = quaternion_rotation_matrix(detection["rotation"])
    rotation_lidar = global_to_lidar[:3, :3] @ rotation_global
    yaw_lidar = math.atan2(rotation_lidar[1, 0], rotation_lidar[0, 0])

    width, length, height = (float(value) for value in detection["size"])
    return {
        "label": str(detection["detection_name"]).lower(),
        "score": float(detection["detection_score"]),
        "bbox_3d": [
            float(translation_lidar[0]),
            float(translation_lidar[1]),
            float(translation_lidar[2]),
            yaw_lidar,
            length,
            width,
            height,
        ],
    }


def main() -> None:
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as input_file:
        results_payload = json.load(input_file)
    with args.metadata_template.open("r", encoding="utf-8") as template_file:
        template_payload = json.load(template_file)

    results = results_payload.get("results")
    template_frames = template_payload.get("frames")
    if not isinstance(results, dict) or not isinstance(template_frames, list):
        raise ValueError("Expected nuScenes 'results' and template 'frames' objects.")

    template_tokens = [str(frame["sample_token"]) for frame in template_frames]
    if set(template_tokens) != set(results):
        raise ValueError(
            "Detection and metadata-template sample tokens differ: "
            f"missing={len(set(template_tokens) - set(results))}, "
            f"extra={len(set(results) - set(template_tokens))}."
        )

    output_frames = []
    for frame in template_frames:
        sample_token = str(frame["sample_token"])
        lidar_to_global = np.asarray(frame["lidar_to_global"], dtype=np.float64)
        if lidar_to_global.shape != (4, 4):
            raise ValueError(f"Invalid lidar_to_global for sample {sample_token}.")
        global_to_lidar = np.linalg.inv(lidar_to_global)
        output_frame = {key: value for key, value in frame.items() if key != "detections"}
        output_frame["detections"] = [
            convert_detection(detection, global_to_lidar)
            for detection in results[sample_token]
        ]
        output_frames.append(output_frame)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump({"frames": output_frames}, output_file)
    detection_count = sum(len(frame["detections"]) for frame in output_frames)
    print(
        f"Wrote {detection_count} detections across {len(output_frames)} chronological frames "
        f"to {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
