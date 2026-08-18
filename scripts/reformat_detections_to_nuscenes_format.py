from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


NUSCENES_ROOT = '/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project/datasets/nuscenes' 
DEFAULT_META = {
	"use_camera": False,
	"use_lidar": True,
	"use_radar": False,
	"use_map": False,
	"use_external": False,
	}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("simpletrack_path",
	    type=Path) 
    return parser.parse_args()

def load_meta():
    return dict(DEFAULT_META)

def load_nuscenes(version, dataroot):
    try:
	from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
	raise ImportError(
		"nuScenes conversion requires nuscenes-devkit to be installed."
		) from exc
	return NuScenes(version=version, dataroot=str(dataroot), verbose=True)

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
    sample_tokens: list[str] 
    for scene in sorted(nusc.scene, key=lambda record: record["name"]):
	if scene["name"] not in selected_scene_names:
	    continue
	sample_token = scene["first_sample_token"]
	while sample_token:
	    sample_tokens.append(sample_token)
	    sample = nusc.get("sample", sample_token)
	    sample_token = sample["next"]
    return sample_tokens

def convert_simpletrack_to_results(
	simpletrack_path: Path,
	meta: dict[str, Any],
	nusc: Any,
	expected_sample_tokens: list[str] | None = None,
	):
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

def main():
    args = parse_args()
    meta = load_meta()
    nusc = load_nuscenes("v1.0-trainval", NUSCENES_ROOT)
    expected_sample_tokens = (
	    load_sample_tokens_from_file(get_eval_sample_tokens(nusc, "val"))
	    )
    results_payload = convert_simpletrack_to_results(
	    simpletrack_path,
	    meta,
	    nusc,
	    expected_sample_tokens=expected_sample_tokens,
	    )
    for sample_token, detections in results_payload["results"].items():
	if len(detections) > 500:
	    detections.sort(key=lambda d: d["detection_score"], reverse=True)
	    results_payload["results"][sample_token] = detections[:500]

    write_results_json(results_payload, output_json)
    detection_count = sum(len(detections) for detections in results_payload["results"].values())
    print(
	    f"Wrote {detection_count} detections across "
	    f"{len(results_payload['results'])} samples to {output_json}"
	    )

    if __name__ == "__main__":
	main()
