# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import torch
from detector.detector import Detector

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class FasterRCNNDetector(Detector):
    """Run Faster R-CNN detection and export MOT-format detections."""

    def __init__(self, input_path, output_path, threshold):
        """Initialize the Faster R-CNN detector.

        Args:
            input_path: Directory containing input image frames.
            output_path: Directory where `det.txt` will be written.
            threshold: Score threshold applied by Detectron2.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.detections = []
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DefaultPredictor(cfg)
    
    def detect(self):
        """Run Faster R-CNN inference on every frame and persist detections.

        Returns:
            list[list[str]] | list[str]: Collected detection lines, either read
            from an existing output file or generated during inference.
        """
        concat_frames, _ = self.read_data()
        frame_index = 1
        output_file = os.path.join(self.output_path, "det.txt")
        with open(output_file, 'a+', encoding='utf-8') as file_obj:
            file_obj.seek(0)
            first_char = file_obj.read(1)
            if first_char:
                print(f"file {output_file} not empty, don't rewrite")
                for line in file_obj:
                    self.detections.append(line)
                return self.detections
            else:
                print(f"file empty running detection on given dataset.")
        
        for concat_frame in concat_frames:
            image = cv2.imread(concat_frame)
            if image is None:
                raise ValueError(f"Unable to read image from path: {concat_frame}")
            detection_results = self.model(image)["instances"].to("cpu")
            lines = self.format_detections(frame_index, detection_results)
            self.detections.append(lines)
            self.write_output(lines)
            frame_index += 1
        return self.detections
    
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
        conf = results.scores.numpy()  # (N,)
        
        lines = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            w = x2 - x1
            h = y2 - y1
            line = f"{frame_index},-1,{x1:.0f},{y1:.0f},{w:.3f},{h:.3f},{c:.6f},-1,-1,-1\n"
            lines.append(line)
        
        return lines
    
    def write_output(self, lines):
        """Append detection lines to the detector output file.

        Args:
            lines: MOT-format lines to append.

        Raises:
            ValueError: If the output directory does not exist.
        """
        out_dir = os.path.dirname(self.output_path) or "."
        if not os.path.exists(out_dir):
            raise ValueError(f"Output directory does not exist: {out_dir}")

        with open(os.path.join(self.output_path, "det.txt"), "a", encoding="utf-8") as f:
            f.writelines(lines)
    
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
       
    def get_model(self):
        """Return the loaded Faster R-CNN predictor."""
        return self.model 
    
