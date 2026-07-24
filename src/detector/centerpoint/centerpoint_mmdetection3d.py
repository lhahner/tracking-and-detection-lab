from pathlib import Path
from detector.detector_mmdetection3d import DetectorMMDetection3D
from definitions import ROOT_DIR
from detector.detector_registry import MODELS

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = (
    Path(ROOT_DIR)
    / "third_party"
    / "mmdetection3d"
    / "configs"
    / "centerpoint"
    / "centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
)
DEFAULT_CHECKPOINT_FILE = (
        PROJECT_DIR
        / "model"
        / "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
)


@MODELS.register("centerpoint_mmdetection3d")
class CenterPointMMDetections3D(DetectorMMDetection3D):
    def __init__(self,
                 dataset,
                 classes,
                 settings,
                 config_file=DEFAULT_CONFIG_FILE,
                 checkpoint_file=DEFAULT_CHECKPOINT_FILE,
                 batch_size=1):
        super().__init__(
                dataset=dataset,
                classes=classes,
                settings=settings,
                config_file=config_file,
                checkpoint_file=checkpoint_file,
                batch_size=batch_size
        ) 
        self.__remove_multisweep_transform_for_preloaded_points()

    def __remove_multisweep_transform_for_preloaded_points(self):
        pipeline = self.model.cfg.test_dataloader.dataset.pipeline
        self.model.cfg.test_dataloader.dataset.pipeline = [
            transform for transform in pipeline
            if "LoadPointsFromMultiSweeps" not in str(transform.get("type"))
        ]
