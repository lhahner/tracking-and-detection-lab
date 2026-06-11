# MMDetection3D metafile references this AMP config name, but the upstream
# repository does not provide a separate file for it. The checkpoint architecture
# matches the non-AMP FPN nuScenes config; AMP only affects training/runtime.
_base_ = './pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
