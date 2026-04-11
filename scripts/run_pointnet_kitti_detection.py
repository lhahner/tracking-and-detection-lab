#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from detector.pointnet.pointnet import PointNetDetector


def main():
    parser = argparse.ArgumentParser(description="Run PointNet-based KITTI LiDAR detection.")
    parser.add_argument("--data-root", required=True, type=str, help="KITTI object detection root.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Fine-tuned PointNet checkpoint.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory for per-frame detection files.")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--use-intensity", action="store_true")
    args = parser.parse_args()

    detector = PointNetDetector(
        input_path=args.data_root,
        output_path=args.output_dir,
        checkpoint_path=args.checkpoint,
        num_points=args.num_points,
        score_threshold=args.score_threshold,
        use_intensity=args.use_intensity,
    )
    detector.detect()


if __name__ == "__main__":
    main()
