import torch
import numpy as np
from detector.detector import Detector
from torch.utils.data import DataLoader, dataloader
from util.file_handler import write_output
from pytorch3d.structures import Pointclouds

if torch.cuda.is_available():
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.structures.bbox_3d import Box3DMode
else:
    exit()
from util.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)
DET_PATH = ""  # TODO

# TODO consider that in Kitti the labels Car and pedestrian are evaluated,
# therefor it would make sense to filter out other in either the format or
# other.


class PointRCNNmmDetections3D(Detector):
    """
    PointRCNN implementation based on the pre-trained
    model from mmdetection3d.

    Attributes:
        :param dataset:
        :param config_file:
        :param checkpoint_file:
        :param classes:
        :param batch_size:
        :param num_inference_samples:
    """
    def __init__(self,
                 dataset, config_file, classes,
                 checkpoint_file="model/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth",
                 batch_size=16, num_inference_samples=50):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_model(self.config_file, self.checkpoint_file)
        self.num_inference_samples = num_inference_samples
        self.classes = classes
        self.batch_size = batch_size

    def detect(self, format_option="kitti") -> list(str):
        """
        Run detection with model using output format for
        further processing.

        Parameters:
            :param format_option:
            :rtype: list(str)

        Returns:
            List of formatted detectiosn that are not empty and and object.
        """
        test_dataloader: torch.utils.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                                             collate_fn=custom_collate)
        formatted_detections: list(str) = []
        for point, sample in test_dataloader:
            num_obj: int = inference_detector(self.model, point)[0].pred_instances_3d.bboxes_3d.tensor.shape[0]
            # Points can contain no  objects
            if num_obj == 0:
                continue
            all_bboxes: torch.tensor = torch.tensor.zeros(self.num_inference_samples, num_obj, 7)
            all_scores: torch.tensor = torch.tensor.zeros(self.num_inference_samples, num_obj)

            for i in enumerate(self.num_inference_samples):
                infered_obj = inference_detector(self.model, point)[0]
                all_bboxes[i] = infered_obj.pred_instances_3d.bboxes_3d.tensor
                all_scores[i] = infered_obj.pred_instances_3d.scores_3d

            bboxes: torch.tensor = all_bboxes.mean(dim=0, keepdim=True)
            scores: torch.tensor = all_scores.mean(dim=0, keepdim=True)
            labels: torch.tensor = inference_detector(self.model, point)[0].pred_instances_3d.labels_3d
            highest_score_index: int = scores.argmax()

            if format_option == "kitti":
                formatted_detections.append(self.format_kitti3d_detections(xyz_centroids=bboxes[highest_score_index, :3],
                                                                           lwh_box=bboxes[highest_score_index, 3:6],
                                                                           yaw=bboxes[6],
                                                                           det_score=scores[highest_score_index],
                                                                           obj_index=labels[highest_score_index]
                                                                           ))
            elif format_option == "sort":
                formatted_detections.append(self.format_sort_detections(frame_index=sample,
                                                                        xyz_centroids=bboxes[highest_score_index, :3],
                                                                        lwh_box=bboxes[highest_score_index, 3:6],
                                                                        yaw=bboxes[6],
                                                                        det_score=scores[highest_score_index]
                                                                        ))

            else:
                raise ValueError("Given format option not supported")

        write_output(f"{DET_PATH}/{format_option}-det.txt", formatted_detections)
        return formatted_detections

    def format_sort_detections(self, frame_index: int, xyz_centroids: np.array, lwh_box: np.array,
                               yaw: torch.tensor, det_score: float) -> str:
        """
        Format detection output string into format that SORT requires to
        work with it.
        Attributes:
            :param frame_index:
            :type frame_index: int
            :param xyz_centroids:
            :type xyz_centroids: np.array
            :param lwh_box:
            :type lwh_box: np.array
            :param yaw:
            :type yaw: torch.tensor
            :param det_score:
            :type det_score: float

        Returns:
            String of shape;
            - x_1: Left boundary coordinate
            - y_1: Top boundary coordinate
            - z_1: Top Left boundary coordinate
            - x_2: Right boundary coordinate
            - y_2: Bottom boundary coordinate
            - z_3: Bottom Right boundary coordinate
            - score: Detection confidence score
        """
        # Rotation matrix based on yaw at corner z in local coordiantes
        rotation_matrix: torch.tensor = torch.tensor([
            [torch.cos(yaw), torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0,                     0,                     1]
            ])
        # compute per bounding each corner of every coodinate,
        # 8 corners meaning 8 cooridnates
        local_box_corner: torch.tensor = torch.tensor([
            [lwh_box[:, 0]/2, lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x left front
            [lwh_box[:, 0]/2, -lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x right front
            [-lwh_box[:, 0]/2, -lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x left back
            [-lwh_box[:, 0]/2, lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x right back
            [lwh_box[:, 0]/2, lwh_box[:, 1]/2, -lwh_box[:, 2]/2],  # corner y left front
            [lwh_box[:, 0]/2, -lwh_box[:, 1]/2, -lwh_box[:, 2]/2],  # corner y right front
            [-lwh_box[:, 0]/2, -lwh_box[:, 1]/2, -lwh_box[:, 2]/2],  # corner y left back
            [-lwh_box[:, 0]/2, lwh_box[:, 1]/2, -lwh_box[:, 2]/2]  # corner y right back
            ])
        # The coordiantes of the bounding box in world coordinates
        world_bbox: torch.tensor = (xyz_centroids + local_box_corner @ rotation_matrix).unsqueeze(0)
        bboxes: torch.tensor = Pointclouds(points=world_bbox).get_bounding_boxes()
        return f"{frame_index},{bboxes.flatten()},{det_score},1,-1,-1,-1"

    def format_kitti3d_detections(self, xyz_centroids: np.array, lwh_box: np.array,
                                  yaw: torch.tensor, det_score: float, obj_index: int) -> str:
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
        obj_type: str = self.classes[obj_index]
        truncated = 0  # always 0 for 3D
        occluded = -1
        alpha = 0
        bbox_2d: np.array = np.array([0, 0, 0, 0])
        dimensions: np.array = np.array([lwh_box[1], lwh_box[0], lwh_box[2]])
        location: np.array = Box3DMode.convert(xyz_centroids, Box3DMode.LIDAR, Box3DMode.CAM)
        rotation_y: float = yaw
        score: float = det_score
        return self.__build_kitti_gt_string(obj_type,
                                            truncated,
                                            occluded,
                                            alpha,
                                            bbox_2d,
                                            dimensions,
                                            location,
                                            rotation_y,
                                            score)

    def __build_kitti_gt_string(
            self,
            obj_type: str,
            truncated: int,
            occluded: float,
            alpha: float,
            bbox_2d: np.array,
            dimesnions: np.array,
            location: np.array,
            rotation_y: float,
            score: float
            ) -> str:
        """
        Format one KITTI 3D detection line.

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
                2D bounding box of object, ranging [-pi..pi]
            bbox_2d:
                2D bounding box of object in the image. Left, top, right bottom pixels.
            dimensions:
                3D object dimensions: height, width, length
            location:
                3D object location x,y,z in camera coordinates
            rotation_y:
                Rotation r_y around Y-axis in camera coordinates.
            score:
                Indicating confidence of values
        Returns:
            Formatted string as Kitti3D evaluation requires.
        """
        return f"{obj_type} {truncated} {occluded} {alpha} {bbox_2d} {dimesnions} {location} {rotation_y} {score}"


@staticmethod
def custom_collate(batch):
    """
    Custom collate function for the provided dataloader
    Parameters:
        :param batch:
    """
    filtered_data = []
    filtered_samples = []
    for item in batch:
        filtered_data.append(item["points"])
        filtered_samples.append(item["sample_id"])
    return filtered_data, filtered_samples
