import torch
import numpy as np
from detector.detector import Detector
from third_party.pointpillars._ext_src.model.pointpillars import PointPillars
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
                 num_inference_samples=5,
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
                results = self.model(batched_pts=points, mode='test')
                labels_reference: torch.tensor = torch.from_numpy(results[0]['labels'])
                num_obj: int = results[0]['lidar_bboxes'].shape[0]
                # Points can contain no  objects
                if num_obj == 0:
                    continue
                all_bboxes, all_scores = self.__sample(point=points, num_obj=num_obj)
                if len(all_bboxes) <= 1 and len(all_scores) <= 1:
                    bboxes_tensor: torch.tensor = torch.from_numpy(all_bboxes[0])
                    scores_tensor: torch.tensor = torch.from_numpy(all_scores[0])
                else:
                    bboxes_tensor: torch.tensor = torch.stack(all_bboxes)
                    scores_tensor: torch.tensor = torch.stack(all_scores)
                if len(all_scores) > 1:
                    scores: torch.tensor = self.__mean_nonzero(tensor=scores_tensor) 
                else:
                    scores: torch.tensor = scores_tensor
                labels: torch.tensor = torch.tensor([3, 1, 2], device=labels_reference.device)[labels_reference]
                if len(all_bboxes) > 1:
                    bboxes: torch.tensor = self.__mean_nonzero(tensor=bboxes_tensor)
                else:
                    bboxes: torch.tensor = bboxes_tensor
                detection_sequence.frames.append(FrameDetection(frame=samples[0],
                                                                highest_score_index=scores.argmax(),
                                                                dets=[
                                                                    Detection(
                                                                        score=score,
                                                                        label=label,
                                                                        box=box)
                                                                    for score, box, label in zip(scores.squeeze(0), bboxes.squeeze(0), labels)
                                                                ],
                                                                targets=targets))
            return detection_sequence

    def __sample(self, point, num_obj):
        all_bboxes: list = []
        all_scores: list = []

        for i in range(0, int(self.num_inference_samples)):
            results = self.model(batched_pts=point, mode='test')
            scores_tensor_infered: torch.tensor = results[0]['scores']
            bboxes_tensor_infered: torch.tensor = results[0]['lidar_bboxes']

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

            all_bboxes.append(torch.from_numpy(bboxes_tensor_infered))
            all_scores.append(torch.from_numpy(scores_tensor_infered))
        return all_bboxes, all_scores
        
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

    def __mean_nonzero(self, tensor: torch.tensor) -> torch.tensor:
        if tensor.numel() <= 0:
            raise ValueError("Tensor to compute only includes zeros.")
        mask: torch.tensor = tensor != 0
        return tensor.sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp(min=1)

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
        filtered_targets.append(item["target"][0])
        filtered_samples.append(item["sample_id"])
    return filtered_data, filtered_targets, filtered_samples
