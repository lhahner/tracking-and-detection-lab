import csv
import gc
import importlib.util
import json
import unittest

from definitions import ROOT_DIR
from pathlib import Path
import torch


def validate_mmdetection3d_integration_environment():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("No GPU skipping test")
    if importlib.util.find_spec("mmdet3d") is None:
        raise unittest.SkipTest("MMDetection3D is not installed")
    if importlib.util.find_spec("nuscenes") is None:
        raise unittest.SkipTest("nuScenes devkit is not installed")


def validate_openpcdet_integration_environment():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("No GPU skipping test")
    if importlib.util.find_spec("pcdet") is None:
        raise unittest.SkipTest("MMDetection3D is not installed")
    if importlib.util.find_spec("nuscenes") is None:
        raise unittest.SkipTest("nuScenes devkit is not installed")


def load_model(url, checkpoint_file, destination=Path(f"{ROOT_DIR}/tests/models/")):
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(f"{destination}/{checkpoint_file}")
    if not checkpoint_path.exists():
        torch.hub.download_url_to_file(
            url=url,
            dst=str(checkpoint_path),
            progress=True,
        )
    return f"{destination}/{checkpoint_file}"


NUSCENES_DETECTION_CSV_FIELDNAMES = [
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
    "velocity_x",
    "velocity_y",
    "attribute_name",
]


def serialize_detections_simple_track_format(detector, detection_path):
    from datasets.nuScenes import DETECTION_CLASSES_BY_INDEX

    detection_path = Path(detection_path)
    frame_count = 0
    detection_count = 0
    first_frame_token = None

    with detection_path.open("w", encoding="utf-8") as output_file:
        output_file.write('{"frames": [')
        for frame_detection in detector.iter_detections():
            if first_frame_token is None:
                first_frame_token = frame_detection.frame

            metadata = frame_detection.metadata
            frame_entry = {
                "frame": frame_count,
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
                detection_count += 1

            if frame_count > 0:
                output_file.write(",")
            json.dump(frame_entry, output_file)
            frame_count += 1

            if frame_count % 50 == 0:
                output_file.flush()
                print(
                    f"wrote {detection_count} SimpleTrack detections from {frame_count} nuScenes frames to {detection_path}",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        output_file.write("]}")

    return {
        "frame_count": frame_count,
        "detection_count": detection_count,
        "first_frame_token": first_frame_token,
        "path": detection_path,
    }


def serialize_detections_csv_format(detector, csv_path, fieldnames=None):
    from datasets.nuScenes import DETECTION_CLASSES_BY_INDEX

    csv_path = Path(csv_path)
    fieldnames = fieldnames or NUSCENES_DETECTION_CSV_FIELDNAMES
    frame_count = 0
    detection_count = 0
    first_frame_token = None

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for frame_detection in detector.iter_detections():
            if first_frame_token is None:
                first_frame_token = frame_detection.frame
            if not isinstance(frame_detection.dets, list):
                raise TypeError("frame_detection.dets must be a list")
            frame_count += 1
            for det in frame_detection.dets:
                box = det.box.detach().cpu() if hasattr(det.box, "detach") else det.box
                label = det.label.item() if hasattr(det.label, "item") else det.label
                score = det.score.item() if hasattr(det.score, "item") else det.score
                velocity_x = box[7] if len(box) > 7 else 0.0
                velocity_y = box[8] if len(box) > 8 else 0.0
                writer.writerow(
                    {
                        "sample_token": str(frame_detection.frame),
                        "detection_name": DETECTION_CLASSES_BY_INDEX[int(label)],
                        "detection_score": round(float(score), 6),
                        "x": round(float(box[0]), 6),
                        "y": round(float(box[1]), 6),
                        "z": round(float(box[2]), 6),
                        "length": round(float(box[3]), 6),
                        "width": round(float(box[4]), 6),
                        "height": round(float(box[5]), 6),
                        "yaw": round(float(box[6]), 6),
                        "velocity_x": round(float(velocity_x), 6),
                        "velocity_y": round(float(velocity_y), 6),
                        "attribute_name": "",
                    }
                )
                detection_count += 1
            if frame_count % 50 == 0:
                output_file.flush()
                print(
                    f"wrote {detection_count} detections from {frame_count} nuScenes frames to {csv_path}",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return {
        "frame_count": frame_count,
        "detection_count": detection_count,
        "first_frame_token": first_frame_token,
        "path": csv_path,
    }
