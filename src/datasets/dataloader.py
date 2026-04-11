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


def get_dataloader(
    dataset,
    batch_size,
    num_points=1024,
    use_intensity=True,
    num_workers=0,
    shuffle=True,
    drop_last=False,
):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=partial(
            object_collate_fn,
            num_points=num_points,
            use_intensity=use_intensity,
        ),
    )
