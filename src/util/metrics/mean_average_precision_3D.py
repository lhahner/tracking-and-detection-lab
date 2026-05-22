import torch
from torchmetrics import Metric
from util.evaluation import Evaluation


class MeanAveragePrecision3D(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("predicted_scores", default=[], dist_reduce_fx="cat")
        self.add_state("ground_truth_labels", default=[], dist_reduce_fx="cat")
        self.classes = kwargs["classes"]
        self.labels = kwargs["labels"]
        self.evaluation = Evaluation()

    def update(self, preds: torch.tensor, target: torch.tensor) -> None:
        preds, target = self._input_format(preds, target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

    def compute(self):
        class_wise_average_precision = []
        for c in self.classes:
            pred_mask = self.labels == c
            gt_mask = self.ground_truth_labels == c

            class_wise_average_precision.append(self.evaluation.compute_average_precision(
                    predicted_detection_scores=self.predicted_scores[pred_mask],
                    ground_truth=self.ground_truth_labels[gt_mask]))

        class_wise_average_precision_tensor = torch.stack(class_wise_average_precision)
        return class_wise_average_precision_tensor.sum() / len(self.classes)
