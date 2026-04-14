from __future__ import annotations

from pathlib import Path

from detector.pointnet.proposals import cluster_to_proposal
from util.kitti_boxes import proposal_iou_bev


def attach_predictions(proposals: list[dict], labels, scores, class_names: dict[int, str], score_threshold: float):
    detections = []
    for proposal, label, score in zip(proposals, labels.tolist(), scores.tolist()):
        if score < score_threshold:
            continue
        class_name = class_names.get(label, "Background")
        if class_name == "Background":
            continue
        detection = dict(proposal)
        detection["label_id"] = label
        detection["label_name"] = class_name
        detection["score"] = float(score)
        detections.append(detection)
    return detections


def non_max_suppression_bev(detections: list[dict], iou_threshold: float = 0.1):
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if all(proposal_iou_bev(detection, existing) < iou_threshold for existing in kept):
            kept.append(detection)
    return kept


def save_kitti_like_detections(output_path: str | Path, detections: list[dict]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for detection in detections:
            center = detection["center"]
            dims = detection["dimensions"]
            yaw = detection["yaw"]
            score = detection["score"]
            label_name = detection["label_name"]
            line = (
                f"{label_name} {score:.6f} "
                f"{center[0]:.4f} {center[1]:.4f} {center[2]:.4f} "
                f"{dims[0]:.4f} {dims[1]:.4f} {dims[2]:.4f} {yaw:.4f}\n"
            )
            f.write(line)
