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
    / "pointpillars"
    / "pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
)
DEFAULT_CHECKPOINT_FILE = (
    PROJECT_DIR
    / "model"
    / "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
)


@MODELS.register("pointpillars_mmdetection3d")
class PointPillarsMMDetections3D(DetectorMMDetection3D):
    def __init__(self,
                 dataset,
                 classes,
                 settings,
                 config_file=DEFAULT_CONFIG_FILE,
                 checkpoint_file=DEFAULT_CHECKPOINT_FILE,
                 batch_size=16):
        super().__init__(
                dataset=dataset,
                classes=classes,
                settings=settings,
                config_file=config_file,
                checkpoint_file=checkpoint_file,
                batch_size=batch_size
        )
