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
    / "ssn"
    / "ssn_hv_regnet-400mf_secfpn_sbn-all_16xb2-2x_nus-3d.py"
)
DEFAULT_CHECKPOINT_FILE = (
    PROJECT_DIR
    / "model"
    / "hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth"
)


@MODELS.register("ssn_mmdetection3d")
class SSNMMDetections3D(DetectorMMDetection3D):
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
