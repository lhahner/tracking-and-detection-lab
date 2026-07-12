import torch
import importlib.util
from definitions import ROOT_DIR
from pathlib import Path

def validate_mmdetection3d_integration_environment():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("No GPU skipping test")
    if importlib.util.find_spec("mmdet3d") is None:
        raise unittest.SkipTest("MMDetection3D is not installed")
    if importlib.util.find_spec("nuscenes") is None:
        raise unittest.SkipTest("nuScenes devkit is not installed")

def load_model(url, checkpoint_file, destination=Path(f"{ROOT_DIR}/tests/models/")):
    
    destination.parent.mkdir(parents=True,
                           exist_ok=True)
    checkpoint_path = Path(f"{destination}/{checkpoint_file}")
    if not checkpoint_path.exists():
        torch.hub.download_url_to_file(
                url=url,
                dst=str(checkpoint_path),
                progress=True
        )
    return f"{destination}/{checkpoint_file}"

#def clean_up(): 
