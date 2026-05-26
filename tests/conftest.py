import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")

for path in (SRC_ROOT, MMDET3D_SRC_ROOT):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
