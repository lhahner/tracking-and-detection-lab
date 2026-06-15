import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_nuscenes_from_csv.py"
spec = importlib.util.spec_from_file_location("evaluate_nuscenes_from_csv", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FakeNuScenes:
    def __init__(self):
        self.scene = [
            {"name": "scene-1", "first_sample_token": "sample-1"},
        ]
        self.samples = {
            "sample-1": {"data": {"LIDAR_TOP": "lidar-top-1"}, "next": "sample-2"},
            "sample-2": {"data": {"LIDAR_TOP": "lidar-top-2"}, "next": ""},
        }
        self.sample_data = {
            "lidar-top-1": {
                "calibrated_sensor_token": "calib-1",
                "ego_pose_token": "ego-1",
            },
            "lidar-top-2": {
                "calibrated_sensor_token": "calib-2",
                "ego_pose_token": "ego-2",
            },
        }
        self.calibrated_sensors = {
            "calib-1": {
                "translation": [1.0, 0.0, 0.0],
                "rotation": [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
            },
            "calib-2": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        }
        self.ego_poses = {
            "ego-1": {
                "translation": [0.0, 2.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            "ego-2": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
        }

    def get(self, table_name, token):
        if table_name == "sample":
            return self.samples[token]
        if table_name == "sample_data":
            return self.sample_data[token]
        if table_name == "calibrated_sensor":
            return self.calibrated_sensors[token]
        if table_name == "ego_pose":
            return self.ego_poses[token]
        raise KeyError(table_name)


def test_convert_row_to_nuscenes_detection_transforms_lidar_box_to_global():
    row = {
        "sample_token": "sample-1",
        "detection_name": "car",
        "detection_score": "0.91",
        "x": "1.0",
        "y": "0.0",
        "z": "0.5",
        "length": "4.0",
        "width": "1.8",
        "height": "1.6",
        "yaw": str(math.pi / 2),
        "velocity_x": "2.0",
        "velocity_y": "0.0",
        "attribute_name": "vehicle.moving",
    }

    resolver = module.NuScenesTransformResolver(FakeNuScenes())
    sample_token, detection = module.convert_row_to_nuscenes_detection(row, 2, resolver)

    assert sample_token == "sample-1"
    np.testing.assert_allclose(detection["translation"], [1.0, 3.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(detection["size"], [1.8, 4.0, 1.6], atol=1e-6)
    np.testing.assert_allclose(detection["rotation"], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(detection["velocity"], [0.0, 2.0], atol=1e-6)
    assert detection["detection_name"] == "car"
    assert detection["detection_score"] == 0.91
    assert detection["attribute_name"] == "vehicle.moving"


def test_load_detection_eval_config_uses_config_factory(monkeypatch):
    calls = []

    def fake_config_factory(name):
        calls.append(name)
        return {"config_name": name}

    common_module = types.ModuleType("nuscenes.eval.common.config")
    common_module.config_factory = fake_config_factory
    monkeypatch.setitem(sys.modules, "nuscenes.eval.common.config", common_module)

    config = module.load_detection_eval_config("detection_cvpr_2019")

    assert config == {"config_name": "detection_cvpr_2019"}
    assert calls == ["detection_cvpr_2019"]


def test_convert_csv_to_results_seeds_empty_eval_samples(tmp_path, monkeypatch):
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "sample_token,detection_name,detection_score,x,y,z,length,width,height,yaw\n"
        "sample-1,car,0.91,1.0,0.0,0.5,4.0,1.8,1.6,0.0\n",
        encoding="utf-8",
    )

    splits_module = types.ModuleType("nuscenes.utils.splits")
    splits_module.create_splits_scenes = lambda verbose=False: {"mini_val": ["scene-1"]}
    monkeypatch.setitem(sys.modules, "nuscenes.utils.splits", splits_module)

    expected_tokens = module.get_eval_sample_tokens(FakeNuScenes(), "mini_val")
    results_payload = module.convert_csv_to_results(
        csv_path,
        module.DEFAULT_META,
        FakeNuScenes(),
        expected_sample_tokens=expected_tokens,
    )

    assert set(results_payload["results"].keys()) == {"sample-1", "sample-2"}
    assert len(results_payload["results"]["sample-1"]) == 1
    assert results_payload["results"]["sample-2"] == []


def test_load_nuscenes_imports_from_nuscenes_submodule(monkeypatch, tmp_path):
    calls = []

    class FakeNuScenesClass:
        def __init__(self, version, dataroot, verbose):
            calls.append((version, dataroot, verbose))

    top_level_module = types.ModuleType("nuscenes")
    submodule = types.ModuleType("nuscenes.nuscenes")
    submodule.NuScenes = FakeNuScenesClass
    monkeypatch.setitem(sys.modules, "nuscenes", top_level_module)
    monkeypatch.setitem(sys.modules, "nuscenes.nuscenes", submodule)

    result = module.load_nuscenes("v1.0-mini", tmp_path)

    assert isinstance(result, FakeNuScenesClass)
    assert calls == [("v1.0-mini", str(tmp_path), True)]
