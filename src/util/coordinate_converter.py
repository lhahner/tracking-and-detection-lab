import numpy as np

class CoordinateConverter:
    """Convert detections between the project's supported box formats."""

    def __init__(self):
        """Initialize a coordinate converter for detection arrays."""
        pass
    
    def convert2DDetectionToBox(self, seq_dets, frame):
        """Convert MOT detections for a frame from `xywh` to `xyxy`.

        Args:
            seq_dets: Detection array containing frame IDs and bounding boxes.
            frame: Frame number to extract.

        Returns:
            numpy.ndarray: Bounding boxes for the frame in `xyxy` format with
            scores preserved in the last column.

        Raises:
            ValueError: If the detections array is empty.
        """
        if len(seq_dets) < 1:
            raise ValueError("The sequence detections are empty")
    
        dets = seq_dets[seq_dets[:,0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]
        return dets
