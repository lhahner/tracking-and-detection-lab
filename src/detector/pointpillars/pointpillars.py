import torch
import numpy as np
from detector.detector import Detector
from third_party.pointpillars._ext_src.model.pointpillars import PointPillars
from third_party.pointpillars._ext_src.dataset import point_range_filter
from torch.utils.data import DataLoader
from entities.detection import DetectionSequence, FrameDetection, Detection
from definitions import ROOT_DIR

CHECKPOINT_FILE = f"{ROOT_DIR}/third_party/pointpillars/_ext_src/pretrained/epoch_160.pth"


class Pointpillars(Detector):
    """
    Wrapper for the Pointpillars model used to detect
    objects in a given point cloud.
    """
    def __init__(self,
                 dataset,
                 config_file,
                 classes,
                 settings,
                 batch_size=16,
                 num_inference_samples=1,
                 checkpoint_file=CHECKPOINT_FILE,
                 device='cpu'):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.num_inference_samples = num_inference_samples
        self.classes = classes
        self.batch_size = batch_size
        self.settings = settings
        self.model = self.__init_model(self.checkpoint_file)

    def detect(self):
        """
        Run inference detection with pre-trained model. This wild iterate
        over the given dataset and run inference on each batch using the 
        custom collate function to gather data as required. Here pointpillars
        also filters the points according to the required point range for the 
        model.

        Returns:
            Object of type DetectionSequence where each detection sequence 
            contains each frame and their accoried detections.
            See src/entities/detection.py for more information.
        """
        test_dataloader: torch.utils.dataloader = DataLoader(dataset=self.dataset,
                                                             batch_size=self.batch_size,
                                                             collate_fn=self.dataset.custom_collate)
        detection_sequence: DetectionSequence = DetectionSequence()
        self.model.eval()
        with torch.no_grad():
            for points, targets, samples in test_dataloader:
                filtered_points = self.__filter_points_for_inference(points)
                results = self.model(batched_pts=filtered_points, mode='test')
                for batch_idx, result in enumerate(results):
                    if len(result) == 0:
                        detection_sequence.frames.append(FrameDetection(
                            frame=samples[batch_idx],
                            highest_score_index=None,
                            dets=[],
                            targets=targets[batch_idx]))
                        continue
                    labels_reference: torch.tensor = torch.from_numpy(result['labels'])

                    scores: torch.tensor = torch.from_numpy(result['scores'])
                    bboxes: torch.tensor = torch.from_numpy(result['lidar_bboxes'])
                    labels: torch.tensor = torch.tensor(self.dataset.labels, 
                                                        device=labels_reference.device)[labels_reference]

                    detection_sequence.frames.append(FrameDetection(
                        frame=samples[batch_idx],
                        highest_score_index=scores.argmax(),
                        dets=[
                            Detection(
                                score=score,
                                label=label,
                                box=box)
                            for score, box, label in zip(scores, bboxes, labels)
                        ],
                        targets=targets[batch_idx]))
            return detection_sequence

    def __filter_points_for_inference(self, points, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
        """
        Filtering the givne points the way pointpillars
        requires it, defaults to a range point range of
        [x_min, y_min, z_min, x_max, y_max, z_max] in meters
        and LIDAR format.

        Params:
            :points the points to be filtered

        Returns:
            The filtered points as a list where each point
            location is batched into one list item.
        """
        filtered_points = []
        for pts in points:
            data_dict = {"pts": pts}
            filtered = point_range_filter(
                data_dict,
                point_range=point_range,
            )
            filtered_points.append(filtered["pts"])
        return filtered_points

    def __init_model(self, checkpoint_file):
        """
        Initalize the model with a given checkpoint file.

        Params:
            :checkpoint_file: the file where the pre-trained
            model is stored.

        Returns:
            The initalized model with the given checkpoint.
        """
        if self.device == 'gpu' and torch.cuda.is_available():
            model = PointPillars(nclasses=len(self.classes)).cuda()
            model.load_state_dict(torch.load(self.checkpoint_file))
            return model
        else:
            model = PointPillars(nclasses=len(self.classes))
            model.load_state_dict(
            torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
            return model
