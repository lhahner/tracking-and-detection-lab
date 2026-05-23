import torch
from torchmetrics import Metric


class MeanAveragePrecision3D(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("predicted_scores", default=[], dist_reduce_fx="cat")
        self.add_state("predicted_labels", default=[], dist_reduce_fx="cat")
        self.add_state("ground_truth_labels", default=[], dist_reduce_fx="cat")
        self.classes = kwargs.get("classes", torch.tensor([], dtype=torch.long))

    def update(self, preds: torch.tensor, target: torch.tensor, labels: torch.tensor) -> None:
        if preds.ndim != 1 or target.ndim != 1 or labels.ndim != 1:
            raise ValueError("preds, target and labels must be one-dimensional tensors")
        if preds.shape != labels.shape:
            raise ValueError("preds and labels must have the same shape")

        self.predicted_scores.append(preds)
        self.predicted_labels.append(labels)
        self.ground_truth_labels.append(target)

    def compute(self):
        if not self.predicted_scores:
            return torch.tensor(0.0)

        predicted_scores = torch.cat(self.predicted_scores)
        predicted_labels = torch.cat(self.predicted_labels)
        ground_truth_labels = torch.cat(self.ground_truth_labels)

        classes = self.classes
        if classes.numel() == 0:
            if predicted_labels.numel() == 0 and ground_truth_labels.numel() == 0:
                return torch.tensor(0.0)
            classes = torch.unique(torch.cat([predicted_labels, ground_truth_labels])) # ?

        class_wise_average_precision = []
        for c in classes:
            pred_mask = predicted_labels == c
            gt_mask = ground_truth_labels == c

            class_predicted_scores = predicted_scores[pred_mask]
            class_ground_truth_labels = ground_truth_labels[gt_mask]

            if class_predicted_scores.numel() == 0 or class_ground_truth_labels.numel() == 0:
                class_wise_average_precision.append(torch.tensor(0.0))
                continue

            sorted_scores, _ = torch.sort(class_predicted_scores, descending=True)
            true_positives = torch.ones_like(sorted_scores) # ? 
            false_positives = torch.zeros_like(sorted_scores) # ?

            cumulative_tp = torch.cumsum(true_positives, dim=0) # ?
            cumulative_fp = torch.cumsum(false_positives, dim=0) # ?

            precisions = cumulative_tp / (cumulative_tp + cumulative_fp) # ?
            recalls = cumulative_tp / class_ground_truth_labels.numel() # ?

            precisions = torch.cat([precisions, torch.tensor([1.0], device=precisions.device)]) # ?
            recalls = torch.cat([recalls, torch.tensor([0.0], device=recalls.device)]) # ?

            class_wise_average_precision.append(
                torch.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
            )

        class_wise_average_precision_tensor = torch.stack(class_wise_average_precision)
        return class_wise_average_precision_tensor.sum() / len(classes)
