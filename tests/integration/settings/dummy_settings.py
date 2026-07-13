from types import SimpleNamespace
from pathlib import Path
from definitions import ROOT_DIR

root_dir = Path(ROOT_DIR)
checkpoint_file = root_dir / "src/detector/centerpoint/model/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
config_file = root_dir / "src/detector/centerpoint/model/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
dataset_path = root_dir / "tests/data/nuScenes_dummy"


def generate_nuscenes_mini_settings_with_custom_detector(detector_name, 
                                                         config_file_path,
                                                         checkpoint_path,
                                                         dataset_name="nuscenes-mini"):
    return SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=f"{ROOT_DIR}/src/detector/{detector_name}/detections/",
                dataset_path=root_dir / "tests/data/nuScenes_dummy",
                checkpoint_path=checkpoint_path,
                config_file=config_file_path,
            ),
            runtime=SimpleNamespace(
                datatype="bin",
                dataset=dataset_name,
                display=False,
            ),
            benchmark=SimpleNamespace(
                iou_threshold=0.4,
                class_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
            tracker=SimpleNamespace(
                max_age=2,
                min_hits=2,
                iou_threshold=0.5
            ),
            dataset=SimpleNamespace(
                classes=[
                    "barrier",
                    "bicycle",
                    "bus",
                    "car",
                    "construction_vehicle",
                    "motorcycle",
                    "pedestrian",
                    "traffic_cone",
                    "trailer",
                    "truck",
                ]
            ),
        )

def generate_nuscenes_mini_settings_with_custom_detector_and_custom_tracker(detector_name, 
                                                                            config_file_path,
                                                                            tracker_name,
                                                                            checkpoint_path,
                                                                            dataset_name="nuscenes-mini"):
    return SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=f"{ROOT_DIR}/src/detector/{detector_name}/detections/",
                dataset_path=root_dir / "tests/data/nuScenes_dummy",
                checkpoint_path=checkpoint_path,
                config_file=config_file_path,
                tracking_path=f"{ROOT_DIR}/src/tracker/{tracker_name}/tracks/{detector_name}_simpletrack.json"
            ),
            runtime=SimpleNamespace(
                datatype="bin",
                dataset="nuscenes-mini",
                display=False,
            ),
            benchmark=SimpleNamespace(
                iou_threshold=0.4,
                class_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(
                classes=[
                    "barrier",
                    "bicycle",
                    "bus",
                    "car",
                    "construction_vehicle",
                    "motorcycle",
                    "pedestrian",
                    "traffic_cone",
                    "trailer",
                    "truck",
                ]
            ),
        )

