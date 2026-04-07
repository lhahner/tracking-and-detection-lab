import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
    }

class Kitti3D(Dataset):
    def __init__(self, data_root, split="train"):
        self.dir_calib = os.path.join(data_root, "training", "calib")
        self.dir_label = os.path.join(data_root, "training", "label_2")
        self.dir_velodyne = os.path.join(data_root, "training", "velodyne")
        self.dir_image = os.path.join(data_root, "training", "image_2")
        self.transform = None
         
        if split == "test":
            self.dir_calib = os.path.join(data_root, "testing", "calib")
            # dir_label = os.path.join(data_root, "testing", "")
            self.dir_velodyne = os.path.join(data_root, "testing", "velodyne")
        
        self.split_file = os.path.join(data_root, "ImageSets", f"{split}.txt")
        with open(self.split_file, "r") as f:
            self.sample_ids = [line.strip() for line in f]
            
        
    def __len__(self):
          return len(self.sample_ids)

    def __getitem__(self, idx):
          sample_id = self.sample_ids[idx]

          image, image_path = self._load_image(sample_id)
          calib, calib_path = self._load_calib(sample_id)
          points, points_path = self._load_velodyne(sample_id)
          target, target_path = self._load_label(sample_id)

          if self.transform is not None:
              image = self.transform(image)

          return {
              "image": image,
              "image_path": image_path,
              "points": points,
              "points_path": points_path,
              "calib": calib,
              "calib_path": calib_path,
              "target": target, 
              "target_path": target_path,
              "sample_id": sample_id
          }

    def _load_image(self, sample_id):
          path = os.path.join(self.dir_image, f"{sample_id}.png")
          return Image.open(path).convert("RGB"), path

    def _load_velodyne(self, sample_id):
          path = os.path.join(self.dir_velodyne, f"{sample_id}.bin")
          if path is None:
              return None
          return np.fromfile(path, dtype=np.float32).reshape(-1, 4), path

    def _load_calib(self, sample_id):
          path = os.path.join(self.dir_calib, f"{sample_id}.txt")
          calib = {}
          with open(path, "r") as f:
              for line in f:
                  key, value = line.strip().split(":", 1)
                  calib[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
          return calib, path

    def _load_label(self, sample_id):
          path = os.path.join(self.dir_label, f"{sample_id}.txt") 
          if path is None:
              return None

          objects = []
          with open(path, "r") as f:
              for line in f:
                  fields = line.strip().split()
                  objects.append({
                      "type": fields[0],
                      "truncated": float(fields[1]),
                      "occluded": int(fields[2]),
                      "alpha": float(fields[3]),
                      "bbox": np.array(fields[4:8], dtype=np.float32),
                      "dimensions": np.array(fields[8:11], dtype=np.float32),  # h, w, l
                      "location": np.array(fields[11:14], dtype=np.float32),   # x, y, z
                      "rotation_y": float(fields[14]),
                  })
          return objects, path
