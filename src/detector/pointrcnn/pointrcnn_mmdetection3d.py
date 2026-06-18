import torch
import os
import numpy as np
from detector.detector import Detector
from torch.utils.data import DataLoader
from entities.detection import Detection, DetectionSequence, FrameDetection
from data_io.serializer import Serializer
from config.settings_loader import SettingsLoader
import torch.nn.functional as Functional

if torch.cuda.is_available():
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.structures.bbox_3d import Box3DMode
else:
    raise EnvironmentError("This model needs a GPU to work")
from config.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


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
                 dataset,
                 config_file,
                 classes,
                 settings,
                 checkpoint_file=f"{PROJECT_DIR}/model/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth",
                 batch_size=16, num_inference_samples=50.
                 ):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_model(self.config_file, self.checkpoint_file)
        self.num_inference_samples = num_inference_samples
        self.classes = classes
        self.batch_size = batch_size
        self.settings = settings
        self.serializer = Serializer(settings=self.settings)

    def detect(self):
        """
        Run detection with model using output format for
        further processing.

        Parameters:
            :param format_option:
            :rtype: list(str)

        Returns:
            List of formatted detectiosn that are not empty and and object.
        """
        test_dataloader: torch.utils.dataloader = DataLoader(dataset=self.dataset, 
                                                             batch_size=self.batch_size,
                                                             collate_fn=self.dataset.custom_collate)
        detection_sequence: DetectionSequence = DetectionSequence()
        for points, targets, samples in test_dataloader:
            for point, target, sample_id in zip(points, targets, samples):
                instance_data_reference = inference_detector(self.model, point)[0][0].pred_instances_3d
                labels_reference: torch.tensor = instance_data_reference.labels_3d
                num_obj: int = instance_data_reference.bboxes_3d.tensor.shape[0]
                # Points can contain no  objects
                if num_obj == 0:
                    continue
                all_bboxes, all_scores = self.__sample(point=point, num_obj=num_obj)

                bboxes_tensor: torch.tensor = torch.stack(all_bboxes)
                scores_tensor: torch.tensor = torch.stack(all_scores)

                scores: torch.tensor = self.__mean_nonzero(tensor=scores_tensor).squeeze(0)
                labels: torch.tensor = torch.tensor([3, 1, 2], device=labels_reference.device)[labels_reference]
                bboxes: torch.tensor = Box3DMode.convert(
                    self.__mean_nonzero(tensor=bboxes_tensor),
                    Box3DMode.LIDAR,
                    Box3DMode.CAM).squeeze(0)

                detection_sequence.frames.append(FrameDetection(frame=sample_id,
                                                                highest_score_index=scores.argmax(),
                                                                dets=[
                                                                    Detection(
                                                                        score=score,
                                                                        label=label,
                                                                        box=box) for score, box, label in zip(scores, bboxes, labels)
                                                                ],
                                                            targets=target))
        return detection_sequence

    def __sample(self, point, num_obj):
        """samples from the model a dedicted number of times.

        :param point:
            The points-set from the dataset.
        :param num_obj:
            The inital number of objects from the very first prediction.
        """
        all_bboxes: list = []
        all_scores: list = []

        for i in range(0, int(self.num_inference_samples)):
            instance_data = inference_detector(self.model, point)[0][0].pred_instances_3d
            scores_tensor_infered: torch.tensor = instance_data.scores_3d
            bboxes_tensor_infered: torch.tensor = instance_data.bboxes_3d.tensor

            num_obj_actual_bbox: int = bboxes_tensor_infered.shape[0]
            num_obj_actual_score: int = scores_tensor_infered.shape[0]

            if num_obj_actual_bbox != num_obj_actual_score:
                raise ValueError("Number of identified objects do not align wiht scores")

            if num_obj_actual_bbox < num_obj:
                shape_diff: int = num_obj - num_obj_actual_bbox
                bboxes_tensor_infered: torch.tensor = Functional.pad(
                                                        input=bboxes_tensor_infered,
                                                        pad=(0, 0, shape_diff, 0),
                                                        mode='constant',
                                                        value=0)
            elif num_obj_actual_bbox > num_obj:
                bboxes_tensor_infered.resize_(num_obj, 7)

            if num_obj_actual_score < num_obj:
                shape_diff: int = num_obj - num_obj_actual_score
                scores_tensor_infered: torch.tensor = Functional.pad(
                                                        input=scores_tensor_infered,
                                                        pad=(0, shape_diff),
                                                        mode='constant',
                                                        value=0)
            elif num_obj_actual_score > num_obj:
                scores_tensor_infered.resize_(num_obj)

            all_bboxes.append(bboxes_tensor_infered)
            all_scores.append(scores_tensor_infered)
        return all_bboxes, all_scores

    def __mean_nonzero(self, tensor: torch.tensor) -> torch.tensor:
        """
        Simple helper function to compute the mean of the given tensor.

        Parameters
            :param tensor:
            :type tensor: torch.tensor
            :rtype: torch.tensor
        """
        if tensor.numel() <= 0:
            raise ValueError("Tensor to compute only includes zeros.")
        mask: torch.tensor = tensor != 0
        return tensor.sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)
