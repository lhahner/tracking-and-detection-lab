# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/iou3d/iou3d_utils.py
"""
WARNING
This code is AI Generated and currently
not verified by human-hand nor written 
any tests.
"""
import torch
# WARNING AI Generated Code start
try:
    from .iou3d_op import (
        boxes_overlap_bev_gpu,
        boxes_iou_bev_gpu,
        nms_gpu,
        nms_normal_gpu as nms_normal_gpu_op,
    )
    _HAS_COMPILED_OPS = True
except ImportError:
    from .iou3d_fallback import (
        boxes_iou_bev as boxes_iou_bev_fallback,
        boxes_overlap_bev as boxes_overlap_bev_fallback,
        nms_normal as nms_normal_fallback,
        rotated_nms,
    )
    _HAS_COMPILED_OPS = False
# AI Generated Code end

def boxes_overlap_bev(boxes_a, boxes_b):
    """Calculate boxes Overlap in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_overlap (torch.Tensor): Overlap result with shape (M, N).
    """
    if not _HAS_COMPILED_OPS:
        return boxes_overlap_bev_fallback(boxes_a, boxes_b)

    ans_overlap = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_overlap)

    return ans_overlap


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    if not _HAS_COMPILED_OPS:
        return boxes_iou_bev_fallback(boxes_a, boxes_b)

    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms_cuda(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    scores = scores[order].contiguous()
    
    if _HAS_COMPILED_OPS:
        keep = torch.zeros(boxes.size(0), dtype=torch.long)
        num_out = nms_gpu(boxes, keep, thresh, boxes.device.index)
        keep = keep[:num_out].to(order.device)
    else:
        keep = rotated_nms(boxes, scores, thresh).to(order.device)

    keep = order[keep].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep

# AI Generated Code Start 
def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()
    scores = scores[order].contiguous()

    if _HAS_COMPILED_OPS:
        keep = torch.zeros(boxes.size(0), dtype=torch.long)
        num_out = nms_normal_gpu_op(
            boxes, keep, thresh, boxes.device.index
        )
        keep = keep[:num_out].to(order.device)
    else:
        keep = nms_normal_fallback(boxes, scores, thresh).to(order.device)

    return order[keep].contiguous()
