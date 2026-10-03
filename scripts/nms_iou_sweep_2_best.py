"""Run class-wise GPU NMS for TransFusion + VoxelNeXt over multiple IoU thresholds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils


DEFAULT_RESULTS_ROOT = Path(
    "/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/"
    "mthesis_lennart_hahner/dir.project/results"
)
DEFAULT_INPUTS = (
    DEFAULT_RESULTS_ROOT
    / "tracking-and-detection-lab-detections/transfusion/"
    "transfusion_openpcdet_nuscenes_detections/"
    "transfusion_openpcdet_nuscenes_detections.json",
    DEFAULT_RESULTS_ROOT
    / "tracking-and-detection-lab-detections/voxelnext/"
    "voxelnext_openpcdet_nuscenes_detections/"
    "voxelnext_openpcdet_nuscenes_detections.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[value / 10 for value in range(1, 10)],
    )
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--max-boxes-per-sample", type=int, default=500)
    parser.add_argument("--transfusion", type=Path, default=DEFAULT_INPUTS[0])
    parser.add_argument("--voxelnext", type=Path, default=DEFAULT_INPUTS[1])
    return parser.parse_args()


def load_frames(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return payload["frames"]


def bbox_to_openpcdet(bbox_3d: list[float]) -> list[float]:
    x, y, z, yaw, length, width, height = bbox_3d[:7]
    return [x, y, z, length, width, height, yaw]


def nms_per_class(detections: list[dict], iou_threshold: float) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for detection in detections:
        by_label[detection["label"]].append(detection)

    fused: list[dict] = []
    for class_detections in by_label.values():
        if len(class_detections) < 2:
            fused.extend(class_detections)
            continue
        boxes = torch.tensor(
            [bbox_to_openpcdet(detection["bbox_3d"]) for detection in class_detections],
            dtype=torch.float32,
            device="cuda",
        )
        scores = torch.tensor(
            [detection["score"] for detection in class_detections],
            dtype=torch.float32,
            device="cuda",
        )
        keep_indices, _ = iou3d_nms_utils.nms_gpu(boxes, scores, iou_threshold)
        fused.extend(class_detections[index] for index in keep_indices.cpu().tolist())
    return fused


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for OpenPCDet rotated NMS.")
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError("All IoU thresholds must be between 0 and 1.")

    detector_frames = {
        "transfusion": load_frames(args.transfusion),
        "voxelnext": load_frames(args.voxelnext),
    }
    detector_by_token = {
        name: {frame["sample_token"]: frame for frame in frames}
        for name, frames in detector_frames.items()
    }
    token_sets = {name: set(frames) for name, frames in detector_by_token.items()}
    reference_token_order = [frame["sample_token"] for frame in detector_frames["transfusion"]]
    reference_tokens = token_sets["transfusion"]
    for name, tokens in token_sets.items():
        if tokens != reference_tokens:
            raise ValueError(
                f"{name} sample tokens differ from TransFusion: "
                f"missing={len(reference_tokens - tokens)}, "
                f"extra={len(tokens - reference_tokens)}"
            )

    combined_by_token: dict[str, list[dict]] = {}
    frame_metadata_by_token: dict[str, dict] = {}
    for token in reference_token_order:
        reference_frame = detector_by_token["transfusion"][token]
        frame_metadata_by_token[token] = {
            key: value for key, value in reference_frame.items() if key != "detections"
        }
        combined_by_token[token] = [
            {
                "label": detection["label"],
                "score": detection["score"],
                "bbox_3d": detection["bbox_3d"],
            }
            for frames_by_token in detector_by_token.values()
            for detection in frames_by_token[token]["detections"]
            if detection["score"] >= args.score_threshold
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for threshold in args.thresholds:
        fused_frames = []
        for token, detections in combined_by_token.items():
            fused = nms_per_class(detections, threshold)
            if len(fused) > args.max_boxes_per_sample:
                fused = sorted(fused, key=lambda item: item["score"], reverse=True)[
                    : args.max_boxes_per_sample
                ]
            output_frame = dict(frame_metadata_by_token[token])
            output_frame["detections"] = fused
            fused_frames.append(output_frame)

        threshold_name = f"{threshold:.1f}".replace(".", "p")
        output_path = args.output_dir / f"nms_2_best_iou_{threshold_name}_detections.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump({"frames": fused_frames}, output_file)
        detection_count = sum(len(frame["detections"]) for frame in fused_frames)
        print(
            f"IoU={threshold:.1f}: wrote {detection_count} detections across "
            f"{len(fused_frames)} frames to {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
