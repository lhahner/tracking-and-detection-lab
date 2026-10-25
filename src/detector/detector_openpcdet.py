import os
from pathlib import Path

from easydict import EasyDict
from detector.detector import Detector
from config.logging_config import LoggingConfig

try:
    from pcdet.config import cfg as global_cfg, cfg_from_yaml_file
    class DetectorOpenPCDet(Detector):
        @staticmethod
        def load_openpcdet_config(config_file):
            config = EasyDict()
            for key in ("ROOT_DIR", "LOCAL_RANK"):
                if key in global_cfg:
                    config[key] = global_cfg[key]
    
            cwd = Path.cwd()
            os.chdir(Path(config_file).resolve().parents[2])
            try:
                cfg_from_yaml_file(str(config_file), config)
            finally:
                os.chdir(cwd)
            return config
except:
    class DetectorOpenPCDet(Detector):
        def __init__(self):
            LoggingConfig().get_logger(__name__).warning("Base detection class for OpenPCDet not available, no OpenPCDet model available.")