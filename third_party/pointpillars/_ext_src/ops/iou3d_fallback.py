"""
WARNING
This code is AI Generated and currently
not verified by human-hand nor written 
any tests.
"""
import cv2
import numpy as np
import torch


def _as_rotated_rect(box):
    x1, y1, x2, y2, angle = box
    center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
    size = (float(x2 - x1), float(y2 - y1))
    angle_degrees = float(-np.degrees(angle))
    return center, size, angle_degrees


def _boxes_as_rotated_rects(boxes):
    return [_as_rotated_rect(box) for box in boxes.detach().cpu().numpy()]


def boxes_overlap_bev(boxes_a, boxes_b):
    rects_a = _boxes_as_rotated_rects(boxes_a)
    rects_b = _boxes_as_rotated_rects(boxes_b)
    overlaps = np.zeros((len(rects_a), len(rects_b)), dtype=np.float32)

    for index_a, rect_a in enumerate(rects_a):
        for index_b, rect_b in enumerate(rects_b):
            _, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
            if intersection is not None:
                overlaps[index_a, index_b] = abs(
                    cv2.contourArea(intersection)
                )

    return torch.as_tensor(
        overlaps,
        dtype=boxes_a.dtype,
        device=boxes_a.device,
    )


def boxes_iou_bev(boxes_a, boxes_b):
    overlaps = boxes_overlap_bev(boxes_a, boxes_b)
    areas_a = (
        (boxes_a[:, 2] - boxes_a[:, 0])
        * (boxes_a[:, 3] - boxes_a[:, 1])
    )
    areas_b = (
        (boxes_b[:, 2] - boxes_b[:, 0])
        * (boxes_b[:, 3] - boxes_b[:, 1])
    )
    return overlaps / (
        areas_a[:, None] + areas_b[None, :] - overlaps + 1e-8
    )


def rotated_nms(boxes, scores, threshold):
    rects = _boxes_as_rotated_rects(boxes)
    scores_array = scores.detach().cpu().float().numpy().tolist()
    if not rects:
        return torch.empty(0, dtype=torch.long)

    min_score = min(scores_array)
    score_threshold = min_score - max(abs(min_score) * 1e-6, 1e-6)
    indices = cv2.dnn.NMSBoxesRotated(
        rects,
        scores_array,
        score_threshold,
        float(threshold),
    )
    return torch.as_tensor(
        np.asarray(indices).reshape(-1),
        dtype=torch.long,
    )


def nms_normal(boxes, scores, threshold):
    boxes_array = boxes.detach().cpu().float().numpy()
    xywh_boxes = np.column_stack((
        boxes_array[:, 0],
        boxes_array[:, 1],
        boxes_array[:, 2] - boxes_array[:, 0],
        boxes_array[:, 3] - boxes_array[:, 1],
    )).tolist()
    scores_array = scores.detach().cpu().float().numpy().tolist()
    if not xywh_boxes:
        return torch.empty(0, dtype=torch.long)
    # AI Generated Code Start
    min_score = min(scores_array)
    # End
    score_threshold = min_score - max(abs(min_score) * 1e-6, 1e-6)
    indices = cv2.dnn.NMSBoxes(
        xywh_boxes,
        scores_array,
        score_threshold,
        float(threshold),
    )
    # AI Generated Code Start
    return torch.as_tensor(
        np.asarray(indices).reshape(-1),
        dtype=torch.long,
    )
    # End
