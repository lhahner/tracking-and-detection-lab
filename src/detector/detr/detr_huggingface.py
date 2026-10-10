from transformers import AutoImageProcessor, DetrForObjectDetection
from detector.detector import Detector
from entities.detection import FrameDetection, Detection, DetectionSequence

import os
import torch
from PIL import Image

class DetrHuggingFace(Detector):
    """Run DETR-based person detection and export MOT-format detections."""

    def __init__(self, input_path, threshold=0.9):
        """Initialize the DETR detector.

        Args:
            input_path: Directory containing input image frames.
            output_path: Directory where `det.txt` will be written.
            threshold: Confidence threshold used during post-processing.
        """
        self.input_path = input_path
        self.threshold = threshold
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.person_label_ids = {
            label_id
            for label_id, label_name in self.model.config.id2label.items()
            if label_name.lower() == "person"
        }
        
    def detect(self):
        """Run DETR inference on every frame and persist MOT detections.

        Returns:
            list[list[str]] | list[str]: Collected detection lines, either read
            from an existing output file or generated during inference.
        """
        concat_frames, _ = self.read_data()
        max_confidence_score = 0
        frames = []
        for frame, concat_frame in enumerate(concat_frames):
            image = Image.open(concat_frame).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            detection_results = self.image_processor.post_process_object_detection(
                outputs=outputs, threshold=self.threshold, target_sizes=target_sizes
            )[0]
            
            formatted_detection_results, max_confidence_score  = self.__format_detections(frame,
                                                                                          detection_results, 
                                                                                          max_confidence_score=max_confidence_score)
            frames.append(FrameDetection(frame=frame,
                                         highest_score_index=max_confidence_score,
                                         dets=formatted_detection_results)
                          )
        return DetectionSequence(frames=frames)
   
    def __format_detections(self, frame_index, results, max_confidence_score):
        """Convert YOLO detections into MOT challenge text lines.

        Args:
            frame_index: One-based frame index.
            results: Ultralytics result object for a single frame.

        Returns:
            list[str]: MOT-format detection lines for person detections only.

        Raises:
            ValueError: If `results` or `frame_index` is missing.
        """
        if results is None:
            raise ValueError("The given results object is None.")
        if frame_index is None:
            raise ValueError("The given frame object is None")
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append(
                Detection(score=score,
                          label=label,
                          box=box)
                )
            if score > max_confidence_score:
                max_confidence_score = score
        return detections, max_confidence_score
    
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
        sorted_frames_concat = [os.path.join(str(self.input_path), frame) for frame in frames]
        return sorted_frames_concat, frames