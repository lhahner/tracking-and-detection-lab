import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from detector.pointnet.preprocess import normalize_points

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

class TestPreprocess(unittest.TestCase):
    def test_normalize_points(self):
      proposal = {
          "center": np.array([0.25, 14.3245, 0.6945], dtype=np.float32),
          "dimensions": np.array([1.888, 0.345, 0.011], dtype=np.float32),
          "yaw": np.float32(0.0),
          "points": np.array([
              [1.1940e+00, 1.4152e+01, 6.8900e-01],
              [1.0390e+00, 1.4182e+01, 6.9000e-01],
              [9.9500e-01, 1.4195e+01, 6.9000e-01],
              [9.5000e-01, 1.4190e+01, 6.9000e-01],
              [7.7400e-01, 1.4239e+01, 6.9100e-01],
              [7.5100e-01, 1.4232e+01, 6.9100e-01],
              [7.0700e-01, 1.4249e+01, 6.9200e-01],
              [6.6200e-01, 1.4243e+01, 6.9100e-01],
              [6.1800e-01, 1.4261e+01, 6.9200e-01],
              [5.7300e-01, 1.4265e+01, 6.9200e-01],
              [5.2900e-01, 1.4286e+01, 6.9300e-01],
              [4.8400e-01, 1.4280e+01, 6.9200e-01],
              [4.6200e-01, 1.4301e+01, 6.9300e-01],
              [4.1700e-01, 1.4300e+01, 6.9300e-01],
              [3.7200e-01, 1.4307e+01, 6.9300e-01],
              [3.2800e-01, 1.4322e+01, 6.9400e-01],
              [2.8200e-01, 1.4313e+01, 6.9300e-01],
              [1.4800e-01, 1.4351e+01, 6.9400e-01],
              [1.2500e-01, 1.4354e+01, 6.9500e-01],
              [8.0000e-02, 1.4358e+01, 6.9500e-01],
              [3.5000e-02, 1.4362e+01, 6.9500e-01],
              [-1.0000e-02, 1.4376e+01, 6.9500e-01],
              [-5.5000e-02, 1.4390e+01, 6.9600e-01],
              [-1.0100e-01, 1.4400e+01, 6.9600e-01],
              [-1.4600e-01, 1.4395e+01, 6.9600e-01],
              [-1.6900e-01, 1.4407e+01, 6.9600e-01],
              [-2.1400e-01, 1.4431e+01, 6.9700e-01],
              [-2.5900e-01, 1.4420e+01, 6.9700e-01],
              [-3.0500e-01, 1.4429e+01, 6.9700e-01],
              [-3.5100e-01, 1.4440e+01, 6.9800e-01],
              [-3.9600e-01, 1.4451e+01, 6.9800e-01],
              [-4.4200e-01, 1.4457e+01, 6.9800e-01],
              [-4.6500e-01, 1.4477e+01, 6.9900e-01],
              [-5.1100e-01, 1.4469e+01, 6.9900e-01],
              [-5.5700e-01, 1.4483e+01, 6.9900e-01],
              [-6.0300e-01, 1.4494e+01, 7.0000e-01],
              [-6.4800e-01, 1.4496e+01, 7.0000e-01],
              [-6.9400e-01, 1.4497e+01, 7.0000e-01],
          ], dtype=np.float32),
      }

      proposal_points = proposal["points"]
      features = normalize_points(proposal_points).max()
      self.assertTrue(features <= 1)
      return None
    
