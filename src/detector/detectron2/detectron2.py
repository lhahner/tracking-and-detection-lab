# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
from detector.detector import Detector

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Detectron2Detector(Detector):
    def __init__(self, input_path, output_path, threshold):
        self.input_path = input_path
        self.output_path = output_path
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.model = DefaultPredictor(cfg)
    
    def detect(self):
        """
        Detect objects using DETR model loaded.
        
        Returns a map of detections, where the index is the frame and the
        value is the corresponding bounding-box and confidence score.
        """
        concat_frames, frames = self.read_data()
        frame_index = 1
        with open(os.path.join(self.output_path, 'det.txt'), 'r') as file_obj:
            first_char = file_obj.read(1)
            if first_char:
                print(f"file {os.path.join(self.output_path, 'det.txt')} not empty, don't rewrite")
                for line in file_obj:
                    self.detections.append(line)
                return self.detections
            else:
                print(f"file empty running detection on given dataset.")
        
        for frame, concat_frame in zip(frames, concat_frames):
            detection_results = predictor(concat_frame)["instances"]
            
            for detection_result in detection_results:
                self.detections.append(self.format_detections(frame_index, detection_result))
                self.write_output(self.format_detections(frame_index, detection_result)) 
            frame_index += 1
        return self.detections
    
    def format_detections(self, frame_index, results):
        """
        Format the detections from Detector2 in the required format so
        that the tracking system is able to handle it.
        We require the following string: <frame-id>,-1,x,y,w,h,confidence,-1,-1,-1
        """
        if results is None:
            raise ValueError("The given results object is None.")
        if frame_index is None:
            raise ValueError("The given frame object is None")
        
        xywh = results.pred_classes  # (N,4)
        conf = results.scores  # (N,)
        
        lines = []
        for (x, y, w, h), c in zip(xywh, conf):
            # If you want ints for x,y like your example, round them.
            # Keep w,h and conf as floats.
            line = f"{frame_index},-1,{x:.0f},{y:.0f},{w:.3f},{h:.3f},{c:.6f},-1,-1,-1\n"
            lines.append(line)
        
        return lines
    
    def write_output(self, lines):
        """
        Append multiple lines to the output file.
        `self.output_path` should be a FILE path, e.g. ".../det.txt"
        """
        out_dir = os.path.dirname(self.output_path) or "."
        if not os.path.exists(out_dir):
            raise ValueError(f"Output directory does not exist: {out_dir}")

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
    
    def read_data(self):
        """
        Read data (frames) from file-system and return them
        as Iteralable object where I can access each frame in detect.
        """
        if not os.path.exists(self.input_path):
           raise ValueError(f"The given input directory {self.input_path} does not exits")
       
        frames = []
        for file_name in os.listdir(self.input_path):
           if file_name.endswith((".png", ".jpg", ".jpeg")):
               frames.append(file_name)
        frames.sort()
        sorted_frames_concat = [self.input_path + frame for frame in frames]
        return sorted_frames_concat, frames
       
    def get_model(self):
        return self.model 
    