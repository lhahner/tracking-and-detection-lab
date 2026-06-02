import np
import dataclasses, json, functools, torch
from util.file_handler import write_output


class Serializer:
    def __init__(self, settings, data_format="json"):
        self.data_format = data_format
        self.settings = settings

    def serialize(self, data):
        """
        Args:
            data: detection sequence to serialize

        Returns:
            The serialization
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
            return json.dumps(data, default=self.__encode_value)
        if self.data_format == "kitti":
            self.format_kitti3d_detections(data)
        if self.data_format == "sort":
            pass

    def format_sort_detections(self, detection_sequence) -> str:
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
        for frame in detection_sequence:
            for det in frame.dets:
                obj_type: str = det.label
                truncated = 0  # always 0 for 3D
                occluded = -1
                alpha = 0
                bbox_2d: np.array = np.array([0, 0, 0, 0])
                location: np.array = det.box[:3].cpu()
                rotation_y: float = det.box[7].item()
                score: float = det.score
                if det.box[3:6].shape == (3,):
                    dimensions: np.array = np.array([
                                        det.box[5].cpu().item(),
                                        det.box[4].cpu().item(),
                                        det.box[6].cpu().item()])
                else:
                    raise IndexError(f"To format the given shape does not match (N, 3) as {det.box[3:6].shape}")
                write_output(self.settings.output_file_path,
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
        bbox_2d_str = " ".join(str(round(i, 2)) for i in bbox_2d)
        dimensions_str = " ".join(str(round(i, 2)) for i in dimensions)
        location_str = " ".join(str(round(i, 2)) for i in location.numpy())
        arr_str: str = " ".join([bbox_2d_str, dimensions_str, location_str])
        return f"{obj_type} {truncated} {occluded} {alpha} {arr_str} {round(rotation_y.item(), 2)} {round(score.item(), 2)}"


    @functools.singledispatch
    def __encode_value(self, data):
        if dataclasses.is_dataclass(data):
            return dataclasses.asdict(data)
        raise ValueError(f"Object of type {type(data).__name__} is not JSON serializable")

    @__encode_value.register(torch.Tensor)
    def _(self, data: torch.Tensor):
        return data.tolist()
