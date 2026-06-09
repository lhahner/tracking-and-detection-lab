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
        test_dataloader: torch.utils.dataloader = DataLoader(dataset=self.dataset,
                                                             batch_size=self.batch_size,
                                                             collate_fn=custom_collate)
        detection_sequence: DetectionSequence = DetectionSequence()
        self.model.eval()
        with torch.no_grad():
            for points, targets, samples in test_dataloader:
                filtered_points = self.__filter_points_for_inference(points)
                results = self.model(batched_pts=filtered_points, mode='test')
                for batch_idx, result in enumerate(results):
                    labels_reference: torch.tensor = torch.from_numpy(result['labels'])
                    num_obj: int = result['lidar_bboxes'].shape[0]
                    if num_obj == 0:
                        continue

                    scores: torch.tensor = torch.from_numpy(result['scores'])
                    bboxes: torch.tensor = torch.from_numpy(result['lidar_bboxes'])
                    labels: torch.tensor = torch.tensor([1, 2, 3], device=labels_reference.device)[labels_reference]

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

    def __filter_points_for_inference(self, points):
        filtered_points = []
        for pts in points:
            data_dict = {"pts": pts}
            filtered = point_range_filter(
                data_dict,
                point_range=[0, -39.68, -3, 69.12, 39.68, 1],
            )
            filtered_points.append(filtered["pts"])
        return filtered_points

    def __init_model(self, checkpoint_file):
        if self.device == 'gpu' and torch.cuda.is_available():
            model = PointPillars(nclasses=len(self.classes)).cuda()
            model.load_state_dict(torch.load(self.checkpoint_file))
            return model
        else:
            model = PointPillars(nclasses=len(self.classes))
            model.load_state_dict(
            torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
            return model


@staticmethod
def custom_collate(batch):
    """
    Custom collate function for the provided dataloader
    Parameters:
        :param batch:
    """
    filtered_data = []
    filtered_targets = []
    filtered_samples = []
    for item in batch:
        filtered_data.append(item["points"])
        filtered_targets.append(item["target"])
        filtered_samples.append(item["sample_id"])
    return filtered_data, filtered_targets, filtered_samples
