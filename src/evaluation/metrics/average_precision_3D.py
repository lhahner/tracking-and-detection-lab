from torchmetrics import Metric
import torch


class AveragePrecision3D(Metric):
    r"""Compute the `Average-Precision (AP) for 3D object detection predictions.

    Args:
        recall:
           Tensor of previously computed recall values.

        precision:
           Tensor of previously computed precision values. 

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``predictions`` or ``ground_truth`` is not of shape (N, 9)
        ValueError:
            If ``predictions`` or ``ground_truth`` is not empty
        

    Example::

        Basic example on how to use this metric internally         

        >>> recall = torch.randn(9)
        >>> precision = torch.randn(2, 9)
        >>> metric = AveragePrecision3D()
        >>> metric.update(
        >>>         recall=recall,
        >>>         precision=ground_truths)
        >>> metric.compute()

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("precision", default=torch.tensor(0), dist_reduce_fx="cat")
        self.add_state("recall", default=torch.tensor(0), dist_reduce_fx="cat")

    def update(self, precision: torch.tensor, recall: torch.tensor) -> None:
        self.precision = precision
        self.recall = recall

    def compute(self):
        if self.precision.numel() == 0 or self.recall.numel() == 0:
            return torch.tensor(0.0, device=self.precision.device)
        # Padding recall to 0 and 1, where 0 is min and 1 is max along the x axis
        padded_recall = torch.cat(
            [
                torch.tensor([0.0], device=self.recall.device),
                self.recall,
                torch.tensor([1.0], device=self.recall.device),
            ]
        )
        # Here the padding is 0 as min but the second there is not max and gets replaced by the max prec.
        padded_precision = torch.cat(
            [
                torch.tensor([0.0], device=self.precision.device),
                self.precision,
                torch.tensor([0.0], device=self.precision.device),
            ]
        )
        # Here we replace the max value with the max value of the precision values y axis
        for i in range(padded_precision.numel() - 2, -1, -1):
            padded_precision[i] = torch.maximum(
                padded_precision[i],
                padded_precision[i + 1],
            )
        # Here we now compute the integration
        # Only the values where have a change since if not we would have width = 0 which results in empty area
        recall_change_indices = torch.where(padded_recall[1:] != padded_recall[:-1])[0]
        # This defines the widht of the rectangles for integration
        padded_recall_diff = padded_recall[recall_change_indices + 1] - padded_recall[recall_change_indices]
        # Here we compute the surface area meaning width_1 * height_2 + ... + width_n * height_n
        ap = torch.sum(padded_recall_diff * padded_precision[recall_change_indices + 1])
        return ap
