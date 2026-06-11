_base_ = [
    '../../../../mmdetection3d/configs/_base_/models/pointpillars_hv_fpn_lyft.py',
    '../../../../mmdetection3d/configs/_base_/datasets/lyft-3d.py', '../../../../mmdetection3d/configs/_base_/schedules/schedule-2x.py',
    '../../../../mmdetection3d/configs/_base_/default_runtime.py'
]
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
