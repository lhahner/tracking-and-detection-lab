from __future__ import annotations

import json
import math
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import io

def _find_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    for candidate in current_file.parents:
        ab3dmot_root = candidate / "third_party" / "AB3DMOT" / "AB3DMOT_libs"
        xinshuo_root = candidate / "third_party" / "Xinshuo_PyToolbox"
        if ab3dmot_root.is_dir() and xinshuo_root.is_dir():
            return candidate
    return current_file.parents[3]


REPO_ROOT = _find_repo_root()
THIRD_PARTY_AB3DMOT_ROOT = REPO_ROOT / "third_party" / "AB3DMOT"
THIRD_PARTY_XINSHUO_TOOLBOX_ROOT = REPO_ROOT / "third_party" / "Xinshuo_PyToolbox"
if str(THIRD_PARTY_AB3DMOT_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_AB3DMOT_ROOT))
if str(THIRD_PARTY_XINSHUO_TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_XINSHUO_TOOLBOX_ROOT))

from definitions import ROOT_DIR
from AB3DMOT_libs import box as ab3dmot_box
from AB3DMOT_libs.io import load_detection
from AB3DMOT_libs.model import AB3DMOT as AB3DMOTModel
from AB3DMOT_libs.utils import Config

from util.quaternion import yaw_to_quaternion, normalize_quaternion, quaternion_multiply, quaternion_rotation_matrix


def __roty(angle: float):
    """Return AB3DMOT's Y-axis rotation matrix without invoking Numba."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


# AB3DMOT's Numba-decorated implementation fails with newer Numba releases
# because its nested lists mix integer literals with floating-point values.
ab3dmot_box.roty = __roty

TRACKING_META = {
    "use_camera": False,
    "use_lidar": True,
    "use_radar": False,
    "use_map": False,
    "use_external": False,
}
MIN_TRACKING_SCORE_BY_LABEL = {
    "bicycle": 0.2,
    "bus": 0.2,
    "car": 0.2,
    "motorcycle": 0.2,
    "pedestrian": 0.2,
    "trailer": 0.2,
    "truck": 0.2,
}
CLASS_NAME_BY_LABEL = {
    "pedestrian": "Pedestrian",
    "car": "Car",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "trailer": "Trailer",
    "truck": "Truck",
}
LABEL_BY_CLASS_NAME = {value: key for key, value in CLASS_NAME_BY_LABEL.items()}
CLASS_ID_BY_CLASS_NAME = {
    "Pedestrian": 1,
    "Car": 2,
    "Bicycle": 3,
    "Motorcycle": 4,
    "Bus": 5,
    "Trailer": 6,
    "Truck": 7,
}
SUPPORTED_PARAM_DET_NAMES = {"centerpoint", "megvii"}
DEFAULT_SEQUENCE_NAME = "default"


class AB3DMOT:
    def __init__(
        self,
        output_path,
        split="val",
        detector_name="centerpoint",
        config_path=None,
        class_map=CLASS_ID_BY_CLASS_NAME,
        class_name_map=CLASS_NAME_BY_LABEL
    ):
        self.output_path = Path(output_path)
        self.split = split
        self.detector_name = detector_name
        default_config = THIRD_PARTY_AB3DMOT_ROOT / "configs" / "nuScenes.yml"
        self.config_path = Path(config_path) if config_path is not None else default_config
        self.root_dir = Path(__file__).resolve().parent
        self.formatted_detection_root = self.root_dir / "formatted_detections"
        self.tracks_root = self.root_dir / "tracks"

        self.class_map = class_map
        self.class_label_map = class_name_map
        self.label_class_map = {value: key for key, value in class_name_map.items()}

        self.__config, _ = Config(str(self.config_path))
        self.__frame_metadata_by_sequence: dict[str, dict[int, dict[str, Any]]] = {}
        self.__frame_order_by_sequence: dict[str, list[int]] = {}
        self.__sequence_names: list[str] = []

    def track(self, detections):
        """
        Run multi-object tracking using detection coming out of one
        object detection module.

        Args:
            detections: Object detections coming from one object detection module.

        Return:
            The tracking results and writes them in required json format to file for nuScenes evaluation
        """
        # Format SimpleTrack Format Detections to AB3DMOT required detections
        self.__reset_runtime_state()
        frame_iterator = self.__iter_simpletrack_frames(Path(detections))
        grouped_lines = self.__format_detections(frame_iterator)
        self.__write_detections(grouped_lines)

        # Process tracking
        tracking_results = self.__track_sequences()
        self.__write_nuscenes_tracking_json(tracking_results, self.output_path)
        return tracking_results

    def __reset_runtime_state(self):
        self.__frame_metadata_by_sequence = {}
        self.__frame_order_by_sequence = {}
        self.__sequence_names = []

    def __iter_simpletrack_frames(self, detection_path):
        """
        Iterate over provided detections and yield every frame
        during call.
        """
        if isinstance(detection_path, (str, Path)):
            with detection_path.open("r", encoding="utf-8") as detection_file:
                frames: dict = json.load(detection_file)
                if not isinstance(frames["frames"], list):
                    raise ValueError(f"{detection_path} must contain a top-level frame list.")
                for frame in frames["frames"]:
                    if not isinstance(frame, dict):
                        raise ValueError(f"Frame entry in {detection_path} is not a JSON object.")
                    yield frame
                return
        else:
            raise ValueError("detection needs to be detection_file")

    def __format_detections(self, frames):
        """
        AB3DMOT requires a different detection format than SimpleTrack,
        therefore here the required format is applied to native SimpleTrack
        formats. Creating files for every sequence/scene to be processed by
        AB3DMOT.

        Args:
            frames: The frames object coming from one of the used detectors.

        Returns:
            Lines containing tracks, grouped by the class
            >>> __format_detections(...)
                {("default", "Car"): [0.0000, -1.0000, -1.0000, ...],
                 ("default", "Bicycle"): [....],
                 ...
                }

        """
        grouped_lines: dict[tuple[str, str], list[str]] = defaultdict(list)
        metadata_by_sequence: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        frame_order_by_sequence: dict[str, list[int]] = defaultdict(list)

        for frame in frames:
            frame_number = int(frame["frame"])
            sequence_name = self.__sequence_name(frame)
            sample_token = str(frame["sample_token"])
            lidar_to_global = np.asarray(frame.get("lidar_to_global", np.eye(4)), dtype=float)
            if lidar_to_global.shape != (4, 4):
                raise ValueError(
                    f"Expected lidar_to_global with shape (4, 4), got {lidar_to_global.shape}"
                )

            metadata_by_sequence[sequence_name][frame_number] = {
                "sample_token": sample_token,
                "frame": frame_number,
                "lidar_to_global": lidar_to_global,
            }
            frame_order_by_sequence[sequence_name].append(frame_number)

            for detection in frame.get("detections", []):
                class_name = self.__class_name(detection)
                if class_name is None:
                    continue
                grouped_lines[(sequence_name, class_name)].append(
                    self.__format_detection_line(frame_number, detection, class_name)
                )

        self.__frame_metadata_by_sequence = {
            sequence_name: dict(sorted(frame_metadata.items()))
            for sequence_name, frame_metadata in metadata_by_sequence.items()
        }
        self.__frame_order_by_sequence = {
            sequence_name: sorted(set(frame_numbers))
            for sequence_name, frame_numbers in frame_order_by_sequence.items()
        }
        self.__sequence_names = sorted(self.__frame_order_by_sequence)
        return grouped_lines

    def __write_detections(self, grouped_lines):
        """
        The detections need to be in a format AB3DMOT can handle.
        Therefore this method expects the formated strings and
        writes them to a file.

        Args:
            grouped_lines which should be in the format defined by format detections
        """
        for class_name in self.class_map:
            target_dir = self.__formatted_category_dir(class_name)
            target_dir.mkdir(parents=True, exist_ok=True)
            for sequence_name in self.__sequence_names:
                lines = grouped_lines.get((sequence_name, class_name), [])
                sequence_file = target_dir / f"{sequence_name}.txt"
                if lines:
                    sequence_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                else:
                    sequence_file.write_text("", encoding="utf-8")

    def __track_sequences(self):
        """
        Wrapper to perform tracking and also flatten the results.

        Returns:
            The flattend object from frame_results
            >>> __track_sequences(...)
                [{'frame': 0,
                  'sample_token': '',
                  'lidar_to_global': [[-0.10, 0.99, 0.02, 250.84],
                                      [-0.99, -0.10, 0.08, 917.47],
                                      ...
                                     ],
                  'tracks': [{
                               'track_id': 31,
                               'bbox_3d': [29.05, 42.89, -0.08, ...],
                               'label': 'car',
                               'score': 0.2,
                               'sample_token': '',
                               'translation': [],
                               'size': []
                             },
                             ...
                            ]
        """
        tracking_results_by_sequence: dict[str, dict[int, dict[str, Any]]] = {}
        flattened_results: list[dict[str, Any]] = []
        id_start = 1

        for sequence_name in self.__sequence_names:
            frame_results = {
                frame_number: {
                    "frame": frame_number,
                    "sample_token": metadata["sample_token"],
                    "lidar_to_global": metadata["lidar_to_global"].tolist(),
                    "tracks": [],
                }
                for frame_number, metadata in self.__frame_metadata_by_sequence[sequence_name].items()
            }

            tracking_classes = [class_name for class_name in self.__config.cat_list if class_name in self.class_map]
            id_start, frame_results = self.__track_sequence_frame_by_class(tracking_classes=tracking_classes,
                                                                           sequence_name=sequence_name,
                                                                           frame_results=frame_results,
                                                                           id_start=id_start)
            tracking_results_by_sequence[sequence_name] = frame_results
            for frame_number in self.__frame_order_by_sequence[sequence_name]:
                flattened_results.append(tracking_results_by_sequence[sequence_name][frame_number])
        return flattened_results

    def __track_sequence_frame_by_class(self, tracking_classes, sequence_name, frame_results, id_start):
        """
        Performs the actual tracking for a given sequence which includes a frame. 
        The tracking is performed class-wise here.

        Args:
            tracking_classes: trackings classes AB3DMOT should run for.
            sequence_name: The sequence to run tracking for.
            frame_results: The DTO to be worked on.
            id_start: Starting id of the sequence, if first 0 else continue where stopped.

        Returns:
            int: Next Id to continue.
            frame_results: The object containing the tracking results
                           >>> self.__track_sequences_frame_by_class(...)
                            (1,
                             {0:
                                {'frame': 0,
                                 'sample_token': '',
                                 'lidar_to_global: '',
                                 'tracks': [{
                                            'track_id': ,
                                            'bbox_3d': ,
                                            'label': ,
                                            'score': ,
                                            'sample_token: ,
                                            'translation': ,
                                            'size': ,
                                            'rotation': ,
                                           }]
                                }
                              })
        """
        for class_name in tracking_classes:
            tracker = self.__create_tracker(class_name, id_start)
            sequence_file = self.__formatted_category_dir(class_name) / f"{sequence_name}.txt"
            dets, has_detections = load_detection(str(sequence_file)) if sequence_file.exists() else ([], False)

            for frame_number in self.__frame_order_by_sequence[sequence_name]:
                dets_frame = self.__frame_detections(dets,
                                                     has_detections,
                                                     frame_number)
                results, _ = tracker.track(dets_frame, frame_number, sequence_name)
                frame_results[frame_number]["tracks"].extend(
                    self.__results_to_tracks(
                        results[0],
                        class_name,
                        self.__frame_metadata_by_sequence[sequence_name][frame_number],
                    )
                )
            id_start = max(id_start, tracker.ID_count[0])
        return id_start, frame_results

    def __create_tracker(self, class_name, id_start):
        cfg = self.__config
        cfg.dataset = "nuScenes"
        cfg.split = self.split
        cfg.det_name = "centerpoint"
        cfg.num_hypo = 1
        cfg.vis = False
        cfg.ego_com = False
        cfg.affi_pro = True
        cfg.score_threshold = -1000
        return AB3DMOTModel(
            cfg,
            class_name,
            calib=None,
            oxts=None,
            img_dir=None,
            vis_dir=None,
            hw=None,
            log=io.StringIO(),
            ID_init=id_start,
        )

    def __frame_detections(self, dets, has_detections, frame_number):
        """
        Filter detections in a frame by the given frame_number, returns
        empty np array of size (0, 7).

        Args:
            dets: All detections coming from the detector.
            has_detections: Mask that provides frames with detections.
            frame_number: Filter for the given frame_number.

        Returns:
            Object containing detetions for the frame and metadata info.
            >>> __frame_detections(...)
                {
                    "dets": [[],
                             [],
                             ...],
                    "info": [[],...]
                }
        """
        if not has_detections or len(dets) == 0:
            return {"dets": np.empty((0, 7), dtype=float), "info": np.empty((0, 7), dtype=float)}

        frame_rows = dets[dets[:, 0] == frame_number]
        if frame_rows.size == 0:
            return {"dets": np.empty((0, 7), dtype=float), "info": np.empty((0, 7), dtype=float)}

        orientation = frame_rows[:, -1].reshape((-1, 1))
        other = frame_rows[:, 1:7]
        info = np.concatenate((orientation, other), axis=1)
        return {"dets": frame_rows[:, 7:14], "info": info}

    def __results_to_tracks(self, results, class_name, frame_metadata):
        """
        Normalizes and transforms the tracking resuts to the required
        format using quaternion transformation. Mainly needed
        because nuScenes requires this format.

        Args:
            results: The results coming from AB3DMOT
            class_name: Class to be transformed
            frame_metadata: Metadata in the frame like sample_metadata

        Returns:
            The normalized tracks for NuScenes evaluation.
        """
        if results.size == 0:
            return []

        normalized_tracks = []
        lidar_to_global = np.asarray(frame_metadata["lidar_to_global"], dtype=float)
        label = self.label_class_map[class_name]
        sensor_rotation = self.__rotation_from_transform(lidar_to_global)

        for row in results:
            height, width, length, x, y, z, yaw = [float(value) for value in row[:7]]
            score = float(row[13])
            if score < MIN_TRACKING_SCORE_BY_LABEL.get(label, 0.0):
                continue
            center = lidar_to_global @ np.asarray([x, y, z, 1.0], dtype=float)
            global_rotation = normalize_quaternion(
                quaternion_multiply(sensor_rotation, yaw_to_quaternion(yaw))
            )
            normalized_tracks.append(
                {
                    "track_id": int(row[7]),
                    "bbox_3d": [x, y, z, yaw, length, width, height, score],
                    "label": label,
                    "score": score,
                    "sample_token": frame_metadata["sample_token"],
                    "translation": [float(center[0]), float(center[1]), float(center[2])],
                    "size": [width, length, height],
                    "rotation": [float(value) for value in global_rotation],
                }
            )
        return normalized_tracks

    def __write_nuscenes_tracking_json(self, tracking_results, output_path):
        """
        Writes the normalized Trackings to json for evaluation in NuScenes.

        Args:
            tracking_results: The AB3DMOT tracking results, normalized
            output_path: The output path of the json containing trackings.
        """
        payload = {"meta": dict(TRACKING_META), "results": {}}
        for frame_result in tracking_results:
            sample_token = str(frame_result["sample_token"])
            payload["results"][sample_token] = []
            for track in frame_result.get("tracks", []):
                payload["results"][sample_token].append(
                    {
                        "sample_token": sample_token,
                        "translation": track["translation"],
                        "size": [float(value) for value in track["size"]],
                        "rotation": [float(value) for value in track["rotation"]],
                        "velocity": [0.0, 0.0],
                        "tracking_id": str(track["track_id"]),
                        "tracking_name": str(track["label"]),
                        "tracking_score": float(track.get("score", 1.0)),
                    }
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __formatted_category_dir(self, class_name):
        """
        Since the detections needed to be reformatted for AB3DMOT to work
        the formatted detections are grouped into classes.

        Args:
            class_name: the class name for which the folder has to be created.

        Returns:
            String name for the folder.
        """
        result_name = f"{self.detector_name}_{class_name}_{self.split}"
        return self.formatted_detection_root / result_name

    def __sequence_name(self, frame_entry):
        """
        Get the sequence name from the frame.

        Args:
            frame_entry: The frame to get the sequence from.

        Returns:
            The sequence string.
        """
        for key in ("scene_name", "scene", "sequence_name", "sequence"):
            value = frame_entry.get(key)
            if value:
                return str(value)
        return DEFAULT_SEQUENCE_NAME

    def __class_name(self, detection):
        """
        Get class name by label from detections.

        Args:
            detection: Detections to get the label from.

        Returns:
            Label as String.
        """
        label = str(detection.get("label", "")).lower()
        return CLASS_NAME_BY_LABEL.get(label)

    def __format_detection_line(self, frame_number, detection, class_name):
        """
        Format detections to a line which AB3DMOT can use.

        Args:
            frame_number: Current frame to format detections.
            detection: The detection to format coming from the detector.
            class_name: The corresponding class_name.

        Returns:
            The formatted detections line
            >>> __format_detection_line(...)
                [0.0, -1.0, -1.0, -1.0, -1.0, 0.27, 1.0, 203.2, 40.1, 41.2, 21.4, 23.3, 0.12, 0.3, 0.4]
        """
        bbox = np.asarray(detection.get("bbox_3d"), dtype=float)
        if bbox.shape[0] < 7:
            raise ValueError("bbox_3d must have at least 7 values: [x,y,z,yaw,l,w,h,(score)]")

        x, y, z, yaw, length, width, height = [float(value) for value in bbox[:7]]
        score = float(detection.get("score", bbox[7] if bbox.shape[0] > 7 else 1.0))
        class_id = CLASS_ID_BY_CLASS_NAME[class_name]

        row = [float(frame_number), -1.0, -1.0, -1.0, -1.0,
               score, float(class_id), height, width, length,
               x, y, z, yaw, yaw]
        return ",".join(f"{value:.6f}" for value in row)

    def __rotation_from_transform(self, transform):
        """
        Defining the quaternion to scale and normalize the tracks for
        nuScenes evaluation.

        Args:
            transform: Inlcudes the rotation matrix needed.

        Return:
            The normaliued quaternion
        """
        rotation_matrix = np.asarray(transform[:3, :3], dtype=np.float64)
        trace = float(np.trace(rotation_matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale,
                    (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale,
                    (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            scale = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale,
                    0.25 * scale,
                    (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale,
                    (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            scale = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale,
                    (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale,
                    0.25 * scale,
                    (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale,
                    (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale,
                    (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
        return normalize_quaternion(quaternion)
