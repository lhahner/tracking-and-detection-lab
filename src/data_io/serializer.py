"""
Basic serialization used to serialize object detection results to drive,
counter part of deserializer.py
"""
import numpy as np
import dataclasses, json, torch, functools, csv, io
from entities.detection import Detection, FrameDetection, DetectionSequence
from data_io.file_handler import write_output
from pathlib import Path
from datasets.nuScenes import DETECTION_CLASSES_BY_INDEX

class Serializer:
    def __init__(self, settings=None, data_format="json", file_name=None):
        self.data_format = data_format
        self.settings = settings
        self.detection_path = self.settings.paths.detection_path
        if file_name is None:
            file_name = f"detections"
        self.file_name = file_name

    def serialize(self, data):
        """
        Args:
            data: detection sequence to serialize, needs to
            be of instance DetectionSequence.
        Returns:
            The serialization depending on the required format
            specified in the class instance, either json, kitti 
            format or other. Similarly the serialization path
            is then defined by the serilization instance.
        Example:
            >>> serialize(detectionSequence)
            >>> [
                {0:
                    [
                        {0.84: [0.1, 0.2, 0.3]},
                        {0.21: [0.4, 0.2, 0.1]}
                    ]
                },
                {1:
                    [
                        {0.20: [0.1, 0.2, 0.3]},
                        {0.21: [0.2, 0.2, 0.3]}
                    ]
                }]
        """
        if self.data_format == "json":
            json_str = json.dumps(data, default=self.__encode_value)
            base = Path(self.detection_path)
            base.mkdir(exist_ok=True)
            jsonpath = base / (self.file_name + ".json")
            jsonpath.write_text(json_str)
            return json_str
        if self.data_format == "kitti" and isinstance(data, DetectionSequence):
            self.format_kitti3d_detections(data)
        if self.data_format == "nuscenes" and isinstance(data, DetectionSequence):
            return self.format_nuScenes_detections(data)
        if self.data_format in {"simple_track", "simpletrack"} and isinstance(data, DetectionSequence):
            return self.format_simple_track_detections(data)
        if self.data_format == "sort" and isinstance(data, DetectionSequence):
            pass

    def format_sort_detections(self, detection_sequence) -> str:
        """TOOD"""
        raise NotImplementedError()
    
    def format_nuScenes_detections(self, detection_sequence) -> str:
        output_buffer = io.StringIO()
        fieldnames = [
            "sample_token",
            "detection_name",
            "detection_score",
            "x",
            "y",
            "z",
            "length",
            "width",
            "height",
            "yaw",
            "velocity_x",
            "velocity_y",
            "attribute_name",
        ]
        writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
        writer.writeheader()

        for frame in detection_sequence.frames:
            sample_token = str(frame.frame)
            for det in frame.dets:
                if det.box.shape[0] < 7:
                    raise IndexError(
                        "nuScenes detections require boxes with at least 7 values: "
                        "[x, y, z, length, width, height, yaw]"
                    )
                label = det.label.item() if hasattr(det.label, "item") else det.label
                detection_name = DETECTION_CLASSES_BY_INDEX[int(label)]
                score = det.score.item() if hasattr(det.score, "item") else det.score
                box = det.box.detach().cpu() if hasattr(det.box, "detach") else det.box
                writer.writerow(
                    {
                        "sample_token": sample_token,
                        "detection_name": detection_name,
                        "detection_score": self.__round_value(score),
                        "x": self.__round_value(box[0]),
                        "y": self.__round_value(box[1]),
                        "z": self.__round_value(box[2]),
                        "length": self.__round_value(box[3]),
                        "width": self.__round_value(box[4]),
                        "height": self.__round_value(box[5]),
                        "yaw": self.__round_value(box[6]),
                        "velocity_x": 0.0,
                        "velocity_y": 0.0,
                        "attribute_name": "",
                    }
                )

        csv_string = output_buffer.getvalue()
        base = Path(self.detection_path)
        base.mkdir(exist_ok=True)
        csv_path = base / (self.file_name + ".csv")
        csv_path.write_text(csv_string, encoding="utf-8")
        return csv_string
    
    def format_simple_track_detections(self, detection_sequence) -> str:
        """
        Parse the data from the DetectionSequence class to the 
        format SimpleTrack requires.
        Parameters:
            :detection_sequence np.array
        Return:
            String with the required format
        Example:
            >>> {
                    "frame": 0,
                    "sample_token": "3e8750f331d7499e9b5123e9eb70f2e2",
                    "time_stamp": 0.0,
                    "lidar_to_global": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]],
                    "ego": [1.0, 0.0, 0.0, 0.0],
                    "aux_info": {
                        "is_key_frame": true
                        },
                    "detections": [
                        {
                            "label": "car",
                            "score": 0.95,
                            "bbox_3d": [
                                635.447,
                                1620.546,
                                -0.326,
                                -0.888442,
                                4.734,
                                2.001,
                                1.481,
                                0.95]
                            }]
                }
        """
        payload = {"frames": []}

        for frame_index, frame in enumerate(detection_sequence.frames):
            metadata = frame.metadata
            if metadata is None:
                raise ValueError("FrameDetection.metadata is required for SimpleTrack serialization")

            frame_entry = {
                "frame": frame_index,
                "sample_token": self.__metadata_value(metadata, "sample_token"),
                "time_stamp": self.__to_float(self.__metadata_value(metadata, "time_stamp")),
                "lidar_to_global": self.__matrix_4x4(
                    self.__metadata_value(metadata, "lidar_to_global"),
                    "lidar_to_global",
                ),
                "ego": self.__matrix_4x4(self.__metadata_value(metadata, "ego"), "ego"),
                "aux_info": dict(self.__metadata_value(metadata, "aux_info")),
                "detections": [],
            }

            for det in frame.dets:
                box = self.__to_numpy(det.box)
                if box.shape[0] < 7:
                    raise IndexError(
                        "SimpleTrack detections require boxes with at least 7 values: "
                        "[x, y, z, length, width, height, yaw]"
                    )

                score = self.__to_float(det.score)
                label = self.__format_detection_label(det.label)
                x, y, z, length, width, height, yaw = box[:7]
                frame_entry["detections"].append(
                    {
                        "label": label,
                        "score": score,
                        "bbox_3d": [
                            float(x),
                            float(y),
                            float(z),
                            float(yaw),
                            float(length),
                            float(width),
                            float(height),
                            score,
                        ],
                    }
                )

            payload["frames"].append(frame_entry)

        json_string = json.dumps(payload, default=self.__encode_value)
        json_path = Path(self.file_name)
        if not json_path.is_absolute():
            json_path = Path(self.detection_path) / json_path
        if json_path.suffix != ".json":
            json_path = json_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_string, encoding="utf-8")
        return json_string

    def format_kitti3d_detections(self, detection_sequence) -> str:
        """
        Parses the data from the mmdetection3d inferecne api to the required
        kitti3d format.
        Parameters:
            :param xyz_centroids:
            :type xyz_centroids: np.array
            :param lwh_box:
            :type lwh_box: np.array
            :param yaw:
            :type yaw: torch.tensor
            :param det_score:
            :type det_score: float
            :param obj_index:
            :type obj_index: int
            :rtype: str
        Returns:
            Formatted string for kitti3D evaluation
        """
        for frame in detection_sequence.frames:
            for det in frame.dets:
                obj_type: str = det.label
                truncated = 0  # always 0 for 3D
                occluded = -1
                alpha = 0
                bbox_2d: np.array = np.array([0, 0, 0, 0])
                location: np.array = det.box[:3].cpu()
                rotation_y: float = det.box[7]
                score: float = det.score
                if det.box[3:6].shape == (3,):
                    dimensions: np.array = np.array([
                                        det.box[5].cpu().item(),
                                        det.box[4].cpu().item(),
                                        det.box[6].cpu().item()])
                else:
                    raise IndexError(f"To format the given shape does not match (N, 3) as {det.box[3:6].shape}")
                det_file = (self.file_name + ".txt")
                write_output((self.detection_path + det_file),
                             self.__build_kitti_gt_string(obj_type=obj_type,
                                                          truncated=truncated,
                                                          occluded=occluded,
                                                          alpha=alpha,
                                                          bbox_2d=bbox_2d,
                                                          dimensions=dimensions,
                                                          location=location,
                                                          rotation_y=rotation_y,
                                                          score=score))

    def __build_kitti_gt_string(
            self,
            obj_type: str,
            truncated: int,
            occluded: float,
            alpha: float,
            bbox_2d: np.array,
            dimensions: np.array,
            location: np.array,
            rotation_y: float,
            score: float
            ) -> str:
        r"""
        Format one KITTI3D detection line.

        KITTI result format:
        type truncated occluded alpha left top right bottom h w l x y z rotation_y score

        Parameters:
            obj_type:
               Types etiher Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare
            truncated:
                Where truncated refers to the object leaving image boundraies.
            occluded:
                (0,1,2,3) indicating occlusion state 0=fully visible,...,3=unknown
            alpha:
                Dim-2D-bounding box of object, ranging [-pi..pi]
            bbox_2d:
                Dim-2D-bounding box of object in the image. Left, top, right bottom pixels.
            dimensions:
                Dim-3D-object dimensions: height, width, length
            location:
                Dim-3D-object location x,y,z in camera coordinates
            rotation_y:
                Rotation r_y around Y-axis in camera coordinates.
            score:
                Indicating confidence of values
        Returns:
            Formatted string as Kitti3D evaluation requires.
        """
        bbox_2d_str = " ".join(str(self.__round_value(i)) for i in bbox_2d)
        dimensions_str = " ".join(str(self.__round_value(i)) for i in dimensions)
        location_str = " ".join(str(self.__round_value(i)) for i in location.numpy())
        arr_str: str = " ".join([bbox_2d_str, dimensions_str, location_str])
        rotation_y_value = rotation_y.item() if hasattr(rotation_y, "item") else rotation_y
        score_value = score.item() if hasattr(score, "item") else score
        return f"{obj_type} {truncated} {occluded} {alpha} {arr_str} {round(rotation_y_value, 2)} {round(score_value, 2)}"

    def __metadata_value(self, metadata, field_name):
        if not hasattr(metadata, field_name):
            raise ValueError(
                f"FrameDetection.metadata must provide {field_name!r} for SimpleTrack serialization"
            )
        value = getattr(metadata, field_name)
        if value is None:
            raise ValueError(
                f"FrameDetection.metadata.{field_name} is required for SimpleTrack serialization"
            )
        return value

    def __matrix_4x4(self, value, field_name):
        matrix = self.__to_numpy(value)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected {field_name} with shape (4, 4), got {matrix.shape}")
        return matrix.astype(float).tolist()

    def __format_detection_label(self, label):
        label = self.__to_scalar(label)
        if isinstance(label, str):
            return label.lower()
        return DETECTION_CLASSES_BY_INDEX[int(label)]

    def __to_numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def __to_scalar(self, value):
        if hasattr(value, "item"):
            return value.item()
        return value

    def __to_float(self, value):
        return float(self.__to_scalar(value))

    def __round_value(self, value):
        if hasattr(value, "item"):
            value = value.item()
        return round(value, 2)
    
    @functools.singledispatchmethod
    def __encode_value(self, data):
        if dataclasses.is_dataclass(data):
            return dataclasses.asdict(data)
        raise ValueError("Dataclass not supported")

    @__encode_value.register(np.ndarray)
    def _(self, data: np.ndarray):
        return data.tolist()

    @__encode_value.register(np.generic)
    def _(self, data: np.generic):
        return data.item()

    @__encode_value.register(torch.Tensor)
    def _(self, data: torch.tensor):
        return data.tolist()
