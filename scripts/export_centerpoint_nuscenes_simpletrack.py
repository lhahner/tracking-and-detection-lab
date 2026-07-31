"""
python scripts/export_centerpoint_nuscenes_simpletrack.py \
        --detector-name pointpillars_mmdetection3d \
        --config path/to/config.py
        --checkpoint path/to/checkpoint.pth
        --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from definitions import ROOT_DIR
from detector.detector_registry import MODELS
from datasets.nuScenes import DETECTION_CLASSES_BY_INDEX
from inference_engine import InferenceEngine


DEFAULT_PROJECT_ROOT = Path(
    "/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project"
)
DEFAULT_CONFIG = (
    Path(ROOT_DIR)
    / "third_party/mmdetection3d/configs/centerpoint"
    / "centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
)
DEFAULT_CHECKPOINT = (
    Path(ROOT_DIR)
    / "tests/models"
    / "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
)
DEFAULT_DETECTOR_NAME = "centerpoint_mmdetection3d"


def default_output_path(detector_name: str, split: str) -> Path:
    return (
        Path(ROOT_DIR)
        / f"src/detector/{detector_name}/detections"
        / f"{detector_name}_nuscenes_simpletrack_{split}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export nuScenes detections from a registered detector in the "
            "SimpleTrack input format with resumable JSONL checkpoints."
        )
    )
    project_root = Path(os.environ.get("PROJECT", DEFAULT_PROJECT_ROOT))
    parser.add_argument("--dataroot", type=Path, default=project_root / "datasets/nuscenes")
    parser.add_argument("--detector-name", default=DEFAULT_DETECTOR_NAME)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Destination SimpleTrack JSON. Defaults to src/detector/<detector-name>/detections/<detector-name>_nuscenes_simpletrack_<split>.json.",
    )
    parser.add_argument(
        "--checkpoint-jsonl",
        type=Path,
        help="Frame-level checkpoint file. Defaults to <output-json>.jsonl.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample tokens already present in the JSONL checkpoint.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing checkpoint instead of resuming or aborting.",
    )
    parser.add_argument(
        "--stop-after-seconds",
        type=int,
        help="Stop cleanly after this many seconds, keeping the checkpoint resumable.",
    )
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()
    if args.output_json is None:
        args.output_json = default_output_path(args.detector_name, args.split)
    return args


def make_settings(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            detection_path=str(args.output_json.parent) + "/",
            dataset_path=args.dataroot,
            checkpoint_path=args.checkpoint,
            config_file=args.config,
        ),
        runtime=SimpleNamespace(
            datatype="bin",
            dataset="nuscenes",
            display=False,
        ),
        benchmark=SimpleNamespace(
            iou_threshold=0.4,
            class_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
        tracker=SimpleNamespace(
            max_age=2,
            min_hits=2,
            iou_threshold=0.5,
        ),
        dataset=SimpleNamespace(
            classes=[
                "barrier",
                "bicycle",
                "bus",
                "car",
                "construction_vehicle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "trailer",
                "truck",
            ]
        ),
    )


def load_completed_tokens(checkpoint_jsonl: Path) -> set[str]:
    completed_tokens: set[str] = set()
    if not checkpoint_jsonl.exists():
        return completed_tokens
    with checkpoint_jsonl.open("r", encoding="utf-8") as checkpoint_file:
        for line_number, line in enumerate(checkpoint_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                frame_entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Checkpoint {checkpoint_jsonl} has invalid JSON on line {line_number}."
                ) from exc
            sample_token = frame_entry.get("sample_token")
            if sample_token:
                completed_tokens.add(str(sample_token))
    return completed_tokens


def frame_detection_to_simpletrack_entry(frame_detection, frame_index: int) -> dict:
    metadata = frame_detection.metadata
    if metadata is None:
        raise ValueError("FrameDetection.metadata is required for SimpleTrack export.")

    frame_entry = {
        "frame": frame_index,
        "sample_token": str(frame_detection.frame),
        "time_stamp": float(metadata.time_stamp),
        "lidar_to_global": metadata.lidar_to_global,
        "ego": metadata.ego,
        "aux_info": dict(metadata.aux_info),
        "detections": [],
    }

    for det in frame_detection.dets:
        box = det.box.detach().cpu() if hasattr(det.box, "detach") else det.box
        label = det.label.item() if hasattr(det.label, "item") else det.label
        score = det.score.item() if hasattr(det.score, "item") else det.score
        frame_entry["detections"].append(
            {
                "label": DETECTION_CLASSES_BY_INDEX[int(label)],
                "score": float(score),
                "bbox_3d": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[6]),
                    float(box[3]),
                    float(box[4]),
                    float(box[5]),
                    float(score),
                ],
            }
        )
    return frame_entry


def assemble_output_json(checkpoint_jsonl: Path, output_json: Path) -> int:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_json.with_suffix(output_json.suffix + ".tmp")
    frame_count = 0
    with checkpoint_jsonl.open("r", encoding="utf-8") as checkpoint_file:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            output_file.write('{"frames": [')
            for line in checkpoint_file:
                line = line.strip()
                if not line:
                    continue
                if frame_count > 0:
                    output_file.write(",")
                output_file.write(line)
                frame_count += 1
            output_file.write("]}")
    temporary_output.replace(output_json)
    return frame_count


def main() -> None:
    args = parse_args()
    checkpoint_jsonl = args.checkpoint_jsonl or args.output_json.with_suffix(
        args.output_json.suffix + ".jsonl"
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_jsonl.exists() and not args.resume and not args.force:
        raise FileExistsError(
            f"{checkpoint_jsonl} already exists. Use --resume to continue or --force to overwrite."
        )
    if args.force and checkpoint_jsonl.exists():
        checkpoint_jsonl.unlink()

    completed_tokens = load_completed_tokens(checkpoint_jsonl) if args.resume else set()
    settings = make_settings(args)
    inference_engine = InferenceEngine(settings=settings)
    dataset = inference_engine.load(
        split=args.split,
        max_samples=args.max_samples,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    expected_frame_count = len(dataset)
    if completed_tokens:
        dataset.sample_records = [
            record
            for record in dataset.sample_records
            if record["sample_token"] not in completed_tokens
        ]

    detector = MODELS.create(
        args.detector_name,
        dataset=dataset,
        config_file=settings.paths.config_file,
        classes=dataset.classes,
        settings=settings,
        checkpoint_file=str(args.checkpoint),
    )

    start_time = time.monotonic()
    written_this_run = 0
    detection_count = 0
    mode = "a" if args.resume and checkpoint_jsonl.exists() else "w"
    with checkpoint_jsonl.open(mode, encoding="utf-8") as checkpoint_file:
        for frame_detection in detector.iter_detections():
            frame_index = len(completed_tokens) + written_this_run
            frame_entry = frame_detection_to_simpletrack_entry(frame_detection, frame_index)
            checkpoint_file.write(json.dumps(frame_entry) + "\n")
            checkpoint_file.flush()
            written_this_run += 1
            detection_count += len(frame_entry["detections"])

            total_done = len(completed_tokens) + written_this_run
            if total_done % args.progress_interval == 0:
                print(
                    f"checkpointed {total_done}/{expected_frame_count} frames "
                    f"({detection_count} detections this run) to {checkpoint_jsonl}",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (
                args.stop_after_seconds is not None
                and time.monotonic() - start_time >= args.stop_after_seconds
            ):
                print("stop-after-seconds reached; checkpoint is resumable.", flush=True)
                break

    total_frames = assemble_output_json(checkpoint_jsonl, args.output_json)
    print(
        f"Wrote {total_frames}/{expected_frame_count} frames to {args.output_json}. "
        f"Checkpoint: {checkpoint_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
