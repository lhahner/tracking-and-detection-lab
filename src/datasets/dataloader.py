from torch.utils.data import DataLoader

"""
Inspired by the implementation of the dataloader of
https://github.com/zhulf0804/PointPillars/blob/main/pointpillars/dataset/dataloader.py
"""
def kitti3d_collate_fn(batch):
      return {
          "image": [sample["image"] for sample in batch],
          "points": [sample["points"] for sample in batch],
          "calib": [sample["calib"] for sample in batch],
          "target": [sample["target"] for sample in batch],
          "sample_id": [sample["sample_id"] for sample in batch]
      }


def get_dataloader(dataset, batch_size, num_workers=0, shuffle=True, drop_last=False):
      return DataLoader(
          dataset=dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          num_workers=num_workers,
          drop_last=drop_last,
          collate_fn=kitti3d_collate_fn,
      )

