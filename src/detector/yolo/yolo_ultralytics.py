import os
import torch

from detector.detector import Detector
from ultralytics import YOLO
from entities.detection import Detection, FrameDetection, DetectionSequence
from pathlib import Path
from definitions import ROOT_DIR


class YoloUltralytics(Detector):
    """Run YOLO-based person detection and export MOT-format detections."""
    def __init__(self, input_path, model):
        """Initialize the YOLO detector.

        Args:
            input_path: Directory containing input image frames.
            output_path: Directory where `det.txt` will be written.
            model_path: Path to the YOLO model weights.
        """
        self.input_path = input_path
        self.model = YOLO(os.path.join(ROOT_DIR, "src", "detector", "yolo", "model", model + ".pt"))
     
    def detect(self):
        """
        Run YOLO inference on every frame and persist MOT detections.

        Returns:
            list[list[str]] | list[str]: Collected detection lines, either read
            from an existing output file or generated during inference.
        """
        concat_frames, frames = self.__read_data()
        frame_index = 1
        frame_detections = []
        for frame, concat_frame in zip(frames, concat_frames):
            detection_results = self.model(concat_frame)
            formatted = []
            max_confidence_score = 0
            for detection_result in detection_results:
                formatted, max_confidence_score = self.__format_detections(frame_index, 
                                                                           detection_result,
                                                                           max_confidence_score=max_confidence_score)
            frame_detections = FrameDetection(highest_score_index=max_confidence_score,
                                              frame=frame,
                                              dets=formatted)
            frame_index += 1
        return DetectionSequence(frames=frame_detections)

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
        
        xyxy = results.boxes.xyxy.cpu().numpy()   # (N,4) -> x1,y1,x2,y2 in original image space
        conf = results.boxes.conf.cpu().numpy()   # (N,)
        cls = results.boxes.cls.cpu().numpy()     # (N,) COCO class ids, person=0

        detections = []
        for (x1, y1, x2, y2), c, class_id in zip(xyxy, conf, cls):
            if int(class_id) != 0:
                continue
                # MOT format expects top-left x,y plus width,height.
            w = x2 - x1
            h = y2 - y1
            detection = Detection(score=conf[0],
                                  label=cls,
                                  box=torch.tensor([x1, y1, x2, y2, w, h])
                                  )
            detections.append(detection)
            if conf[0] > max_confidence_score:
                max_confidence_score = conf[0]
        return detections, max_confidence_score
    
    def __read_data(self):
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