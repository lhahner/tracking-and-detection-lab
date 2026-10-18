from pathlib import Path

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from detector.detector import Detector
from src.registry import MODELS
from entities.detection import Detection, DetectionSequence, FrameDetection
from config.logging_config import LoggingConfig, logged
from definitions import ROOT_DIR


@logged
class DetectorDetectron2(Detector):

    def detect(self):
        """Run Faster R-CNN inference on every frame and persist detections.

        Returns:
            list[list[str]] | list[str]: Collected detection lines, either read
            from an existing output file or generated during inference.
        """
        concat_frames, _ = self.read_data()
        frames = []
        max_confidence_score = 0
        for frame, concat_frame in enumerate(concat_frames):
            image = cv2.imread(concat_frame)
            if image is None:
                raise ValueError(f"Unable to read image from path: {concat_frame}")
            detection_results = self.model(image)["instances"].to("cpu")
            formatted_detection_results = self.format_detections(frame, detection_results)
            frames.append(FrameDetection(frame=frame,
                                         highest_score_index=max_confidence_score,
                                         dets=formatted_detection_results,
                                         )
                          )
        return DetectionSequence(frames=frames)
    
    def format_detections(self, frame_index, results):
        """Convert Detectron2 detections into MOT challenge text lines.

        Args:
            frame_index: One-based frame index.
            results: Detectron2 `Instances` object for a single frame.

        Returns:
            list[str]: MOT-format detection lines.

        Raises:
            ValueError: If `results` or `frame_index` is missing.
        """
        if results is None:
            raise ValueError("The given results object is None.")
        if frame_index is None:
            raise ValueError("The given frame object is None")
        
        xyxy = results.pred_boxes.tensor.numpy()  # (N,4): x1,y1,x2,y2
        conf_scores = results.scores.numpy()  # (N,)
        pred_classes = results.pred_classes.numpy()
         
        detections = []
        for (x1, y1, x2, y2), conf, pred_class in zip(xyxy, conf_scores, pred_classes):
            w = x2 - x1
            h = y2 - y1
            detections.append(Detection(score=conf,
                                        label=pred_class,
                                        box=torch.tensor([x1, y1, x2, y2, w, h])
                                        ))
        
        return detections
    
    def read_data(self):
        """Read and sort image frame paths from the input directory.

        Returns:
            tuple[list[str], list[str]]: Absolute frame paths and corresponding
            frame filenames.

        Raises:
            ValueError: If the input directory does not exist.
        """
        if not os.path.exists(self.input_path):
           raise ValueError(f"The given input directory {self.input_path} does not exits")
       
        frames = []
        for file_name in os.listdir(self.input_path):
           if file_name.endswith((".png", ".jpg", ".jpeg")):
               frames.append(file_name)
        frames.sort()
        sorted_frames_concat = [os.path.join(self.input_path, frame) for frame in frames]
        return sorted_frames_concat, frames
