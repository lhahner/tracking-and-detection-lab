"""
Basic serialization used to serialize object detection results to drive,
counter part of deserializer.py
"""
import numpy as np
import dataclasses, json, torch, functools
import datetime
from entities.detection import Detection, FrameDetection, DetectionSequence
from util.file_handler import write_output
from pathlib import Path

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
        if self.data_format == "sort" and isinstance(data, DetectionSequence):
            pass

    def format_sort_detections(self, detection_sequence) -> str:
        """TOOD"""
        raise NotImplementedError()

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

    def __round_value(self, value):
        if hasattr(value, "item"):
            value = value.item()
        return round(value, 2)
    
    @functools.singledispatchmethod
    def __encode_value(self, data):
        if dataclasses.is_dataclass(data):
            return dataclasses.asdict(data)
        raise ValueError("Dataclass not supported")

    @__encode_value.register(torch.Tensor)
    def _(self, data: torch.tensor):
        return data.tolist()
