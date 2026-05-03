import importlib.util
import os
import sys
import unittest
from pathlib import Path
TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MMDET3D_SRC_ROOT not in sys.path:
    sys.path.insert(0, MMDET3D_SRC_ROOT)


class TestMMDetection3DCompatability(unittest.TestCase):
    def test_cpu_only_compatability(self):
        if importlib.util.find_spec("mmdet3d") is None:
            self.skipTest("mmdet3d is not installed.")

        elif os.path.exists(os.path.join(PROJECT_ROOT, "external/mmdetection3d-cpu-only")):
            MMDET3D = os.path.join(PROJECT_ROOT,
                                   "external/mmdetection3d-cpu-only")
            config_file = os.path.join(MMDET3D,
                                       "configs/pointnet2/"
                                       "pointnet2_msg_2xb16-"
                                       "cosine-250e_scannet-seg.py")
            checkpoint_file = Path(str(
                f"{PROJECT_ROOT}/tests/data"
                "/pointnet2_msg_xyz-only_16x2_"
                "cosine_250e_scannet_seg-3d-"
                "20class_20210514_143838-b4a3cf89_dummy.pth")
            )
        else:
            config_file = Path(str('pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'))
            checkpoint_file = Path(str('hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'))

        if not config_file or not checkpoint_file:
            self.skipTest("Config or Checkpoint path not present")

        config_path = Path(config_file).expanduser()
        checkpoint_path = Path(checkpoint_file).expanduser()
        sample_path = (
            Path(PROJECT_ROOT) / "tests/data/kitti3d_dummy/"
                                 "training/velodyne/000000.bin"
        )

        if not checkpoint_path.exists():
            raise ValueError(
                "MMDET3D checkpoint file not found: "
                f"{checkpoint_path}"
            )
        if not sample_path.exists():
            raise ValueError(
                "Point-cloud sample file not found:"
                f"{sample_path}"
            )

        from mmdet3d.apis import inference_detector, init_model
        import mmdet3d.models.backbones.pointnet2_sa_msg  # noqa: F401
        import mmdet3d.models.data_preprocessors.data_preprocessor  # noqa: F401
        import mmdet3d.models.decode_heads.pointnet2_head  # noqa: F401
        import mmdet3d.models.segmentors.encoder_decoder  # noqa: F401
        import mmdet3d.datasets.transforms.formating  # noqa: F401
        import mmdet3d.datasets.transforms.loading  # noqa: F401

        model = init_model(
            str(config_path),
            str(checkpoint_path),
            device="cpu",
            cfg_options={
                "test_dataloader.dataset.pipeline": [
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="DEPTH",
                        shift_height=False,
                        use_color=True,
                        load_dim=6,
                        use_dim=[0, 1, 2, 3, 4, 5],
                        backend_args=None,
                    ),
                    dict(type="NormalizePointsColor", color_mean=None),
                    dict(type="Pack3DDetInputs", keys=["points"]),
                ]
            },
        )
        self.assertIsNotNone(model)
