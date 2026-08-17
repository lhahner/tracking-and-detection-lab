import json
from collections import defaultdict
import torch

try:
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
except ImportError as exc:
    raise ImportError(
        "This script requires OpenPCDet's compiled iou3d_nms_utils (GPU rotated-NMS CUDA "
        "ops). This should already be built in track-lab since SECOND/TransFusion/VoxelNeXt "
        "run through OpenPCDet. If the import fails, verify with "
        "'python -c \"from pcdet.ops.iou3d_nms import iou3d_nms_cuda\"' inside the env."
    ) from exc

if not torch.cuda.is_available():
    raise RuntimeError(
        "iou3d_nms_utils.nms_gpu requires a CUDA device — no GPU visible. "
        "Check that this job was actually scheduled with --gpus=A100:1."
    )
DEVICE = "cuda"

NUSCENES_ROOT = '/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project/datasets/nuscenes'

SECOND_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/second/second_openpcdet_nuscenes_detections/second_openpcdet_nuscenes_detections.json'
TRANSFUSION_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/transfusion/transfusion_openpcdet_nuscenes_detections/transfusion_openpcdet_nuscenes_detections.json'
VOXELNEXT_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/voxelnext/voxelnext_openpcdet_nuscenes_detections/voxelnext_openpcdet_nuscenes_detections.json'
POINTPILLARS_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/pointpillars/pointpillars_mmdetection3d_nuscenes_detections/pointpillars_mmdetection3d_nuscenes_detections.json'
CENTERPOINT_VOXELNET_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/centerpoint_voxelnet/centerpoint_voxelnet_mmdetection3d_nuscenes_detections/centerpoint_voxelnet_mmdetection3d_nuscenes_detections.json'
SSN_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/ssn/ssn_mmdetection3d_nuscenes_detections/ssn_mmdetection3d_nuscenes_detections.json'
REGNET_DETECTIONS_PATH = '/user/lennart.hahner/u28856/tracking-and-detection-lab-detections/regnet/regnet_mmdetection3d_nuscenes_detections/regnet_mmdetection3d_nuscenes_detections.json'

OUTPUT_JSON = '/user/lennart.hahner/u28856/jupyterhub-gwdg/deep-learning-based-detection-systems-for-multi-object-tracking-in-lidar-data/notebooks/scripts/data/nms_fused_detections.json'

SCORE_THRESHOLD = 0.1       # tune this — drop obviously-noisy low-confidence boxes
MAX_BOXES_PER_SAMPLE = 500  # matches nuScenes detection eval's hard limit

def load_frames(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data["frames"]

detector_frames = {
    "second": load_frames(SECOND_DETECTIONS_PATH),
    "transfusion": load_frames(TRANSFUSION_DETECTIONS_PATH),
    "voxelnext": load_frames(VOXELNEXT_DETECTIONS_PATH),
    "pointpillars": load_frames(POINTPILLARS_DETECTIONS_PATH),
    "centerpoint_voxelnet": load_frames(CENTERPOINT_VOXELNET_DETECTIONS_PATH),
    "ssn": load_frames(SSN_DETECTIONS_PATH),
}

detector_by_token = {
    name: {frame["sample_token"]: frame for frame in frames}
    for name, frames in detector_frames.items()
}

token_sets = {name: set(d.keys()) for name, d in detector_by_token.items()}
reference_tokens = token_sets["second"]
for name, tokens in token_sets.items():
    if tokens != reference_tokens:
        print(f"WARNING: {name} differs from 'second' — "
              f"missing {len(reference_tokens - tokens)}, extra {len(tokens - reference_tokens)}")

all_sample_tokens = sorted(reference_tokens)

def bbox_to_openpcdet(bbox_3d):
    # stored order: x, y, z, yaw, length, width, height, (score)
    # OpenPCDet nms_gpu order: x, y, z, dx, dy, dz, heading
    x, y, z, yaw, length, width, height = bbox_3d[:7]
    return [x, y, z, length, width, height, yaw]

def nms_3d_gpu(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []
    if len(detections) == 1:
        return detections

    boxes = torch.tensor(
        [bbox_to_openpcdet(d["bbox_3d"]) for d in detections],
        dtype=torch.float32, device=DEVICE,
    )
    scores = torch.tensor(
        [d["score"] for d in detections],
        dtype=torch.float32, device=DEVICE,
    )
    keep_idx, _ = iou3d_nms_utils.nms_gpu(boxes, scores, iou_threshold)
    keep_idx = keep_idx.cpu().tolist()
    return [detections[i] for i in keep_idx]

def nms_per_class(detections, iou_threshold=0.5):
    """Run NMS independently within each class label to avoid cross-class suppression."""
    by_label = defaultdict(list)
    for det in detections:
        by_label[det["label"]].append(det)
    fused = []
    for dets in by_label.values():
        fused.extend(nms_3d_gpu(dets, iou_threshold=iou_threshold))
    return fused

fused_frames = []
for token in all_sample_tokens:
    combined_detections = []
    for name, by_token in detector_by_token.items():
        frame = by_token.get(token)
        if frame is None:
            continue
        for det in frame["detections"]:
            if det["score"] < SCORE_THRESHOLD:
                continue
            combined_detections.append({
                "label": det["label"],
                "score": det["score"],
                "bbox_3d": det["bbox_3d"],
            })
    fused_detections = nms_per_class(combined_detections, iou_threshold=0.5)
    if len(fused_detections) > MAX_BOXES_PER_SAMPLE:
        fused_detections = sorted(fused_detections, key=lambda d: d["score"], reverse=True)[:MAX_BOXES_PER_SAMPLE]
    fused_frames.append({
        "sample_token": token,
        "detections": fused_detections,
    })

output_payload = {"frames": fused_frames}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_payload, f)

total_in = sum(len(frame["detections"]) for by_token in detector_by_token.values() for frame in by_token.values())
total_out = sum(len(frame["detections"]) for frame in fused_frames)
print(f"Wrote {len(fused_frames)} fused frames to {OUTPUT_JSON}")
