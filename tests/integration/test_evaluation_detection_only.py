import os
import sys
import torch
import unittest
import torch
from pathlib import Path

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MMDET3D_SRC_ROOT not in sys.path:
    sys.path.insert(0, MMDET3D_SRC_ROOT)

from util.evaluation import Evaluation


class TestEvaluationOnly(unittest.TestCase):
