from functools import partial

import torch
from torch.utils.data import DataLoader

from detector.pointnet.preprocess import prepare_crop


def object_collate_fn(batch, num_points=1024, use_intensity=True):
    points = torch.stack(
        [
            prepare_crop(
                item["raw_points"],
                num_points=num_points,
                use_intensity=use_intensity,
            )
            for item in batch
        ],
        dim=0,
    )
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"points": points, "labels": labels}


def frame_collate_fn(batch):
    return {
        "image": [item["image"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "points": [item["points"] for item in batch],
        "points_path": [item["points_path"] for item in batch],
        "calib": [item["calib"] for item in batch],
        "calib_path": [item["calib_path"] for item in batch],
        "target": [item["target"] for item in batch],
        "target_path": [item["target_path"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }


def get_dataloader(
    dataset,
    batch_size,
    num_points=1024,
    use_intensity=True,
    num_workers=0,
    shuffle=True,
    drop_last=False,
    collate_mode="auto",
    logger=None
):
    if logger is None:
        raise ValueError("To load data provide logger")
    
    if collate_mode == "auto":
        collate_mode = getattr(dataset, "mode", "frame")

    if collate_mode == "object":
        self.logger.info("Loading;" 
                         f"collate_mode={collate_mode}"
                         f"num_points={num_points}"
                         f"batch_size={batch_size}")
        collate_fn = partial(
            object_collate_fn,
            num_points=num_points,
            use_intensity=use_intensity,
        )
    elif collate_mode == "frame":
        self.logger.info("Loading;" 
                         f"collate_mode={collate_mode}"
                         f"num_points={num_points}"
                         f"batch_size={batch_size}")
        collate_fn = frame_collate_fn
    else:
        raise ValueError(
            f"Unsupported collate_mode '{collate_mode}'. "
            "Expected 'auto', 'frame', or 'object'."
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

