import os
from pathlib import Path

from easydict import EasyDict
from pcdet.config import cfg as global_cfg, cfg_from_yaml_file


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
