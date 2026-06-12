import math

import numpy as np
import pytest
import torch

from datasets.nuScenes import NuScenesDataset


def test_transform_matrix_applies_translation_and_z_rotation():
    quaternion = [
        math.cos(math.pi / 4),
        0.0,
        0.0,
        math.sin(math.pi / 4),
    ]
    transform = NuScenesDataset._transform_matrix([1.0, 2.0, 3.0], quaternion)

    transformed = NuScenesDataset._transform_points(
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        transform,
    )

    np.testing.assert_allclose(transformed, [[1.0, 3.0, 3.0]], atol=1e-6)


def test_quaternion_rotation_matrix_rejects_zero_quaternion():
    with pytest.raises(ValueError, match="non-zero norm"):
        NuScenesDataset._quaternion_rotation_matrix([0.0, 0.0, 0.0, 0.0])


def test_transform_points_rejects_invalid_shape():
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        NuScenesDataset._transform_points(
            np.zeros((2, 4), dtype=np.float32),
            np.eye(4),
        )


def test_build_sample_index_preserves_scene_boundaries():
    dataset = NuScenesDataset.__new__(NuScenesDataset)
    dataset.nusc = FakeNuScenes()

    records = dataset._build_sample_index(["scene-0001"])

    assert [record["sample_token"] for record in records] == ["sample-1", "sample-2"]
    assert records[0]["is_first_frame"] is True
    assert records[0]["is_last_frame"] is False
    assert records[1]["is_first_frame"] is False
    assert records[1]["is_last_frame"] is True


def test_convert_ground_truth_uses_internal_box_layout():
    dataset = NuScenesDataset.__new__(NuScenesDataset)
    targets = [
        {
            "box": np.asarray([1, 2, 3, 4, 5, 6, 0.5], dtype=np.float32),
            "label": 4,
        }
    ]

    converted = dataset.convert_ground_truth(targets, frame="000001")

    assert converted.shape == (1, 9)
    assert converted.dtype == torch.float32
    torch.testing.assert_close(
        converted[0],
        torch.tensor([1, 2, 3, 4, 5, 6, 0.5, 0, 4], dtype=torch.float32),
    )


def test_validate_split_version_rejects_mini_trainval_mix():
    dataset = NuScenesDataset.__new__(NuScenesDataset)
    dataset.version = "v1.0-mini"
    dataset.split = "val"

    with pytest.raises(ValueError, match="not compatible"):
        dataset._validate_split_version()


class FakeNuScenes:
    def __init__(self):
        self.scene = [
            {
                "token": "scene-token-1",
                "name": "scene-0001",
                "first_sample_token": "sample-1",
            },
            {
                "token": "scene-token-2",
                "name": "scene-9999",
                "first_sample_token": "other-sample",
            },
        ]
        self.samples = {
            "sample-1": {
                "token": "sample-1",
                "next": "sample-2",
            },
            "sample-2": {
                "token": "sample-2",
                "next": "",
            },
        }

    def get(self, table_name, token):
        assert table_name == "sample"
        return self.samples[token]
