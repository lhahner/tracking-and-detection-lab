import torch
import importlib.util
import os
import unittest
import shutil

from definitions import ROOT_DIR
from pathlib import Path

def validate_mmdetection3d_integration_environment():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("No GPU skipping test")
    if importlib.util.find_spec("mmdet3d") is None:
        raise unittest.SkipTest("MMDetection3D is not installed")
    if importlib.util.find_spec("nuscenes") is None:
        raise unittest.SkipTest("nuScenes devkit is not installed")

def validate_openpcdet_integration_environment():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("No GPU skipping test")
    if importlib.util.find_spec("pcdet") is None:
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

from pathlib import Path


def create_dummy_plugin():
    plugin_path = Path(ROOT_DIR) / "plugins" / "dummy_plugin"
    module_path = plugin_path / "ai_wrapper_mistral"

    module_path.mkdir(parents=True, exist_ok=True)

    plugin_config = """
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-wrapper-mistral"
version = "0.0.1"

[project.entry-points."ai_wrapper.models"]
mistral = "ai_wrapper_mistral.model:MistralModel"

[tool]
type = "detector"
""".strip()

    (plugin_path / "pyproject.toml").write_text(
        plugin_config,
        encoding="utf-8"
    )

    (module_path / "__init__.py").write_text(
        "",
        encoding="utf-8"
    )

    model_class = """
class MistralModel:
    pass
""".strip()

    (module_path / "model.py").write_text(
        model_class,
        encoding="utf-8"
    )

    return plugin_path


def clean_up_dummy_folder(path):
    shutil.rmtree(path)
