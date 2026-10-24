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
from detector.detector_detectron2 import DetectorDetectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


@MODELS.register("mask_faster_rcnn_detectron2")
class MaskFasterRCNNDetectron2Detector(DetectorDetectron2):
    """Run Mask R-CNN detection and export MOT-format detections."""

    def __init__(self, input_path, output_path, threshold):
        """Initialize the Mask R-CNN detector.

        Args:
            input_path: Directory containing input image frames.
            output_path: Directory where `det.txt` will be written.
            threshold: Score threshold applied by Detectron2.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.detections = []
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DefaultPredictor(cfg)
    