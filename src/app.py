from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

from util.datatype import Datatype
from util.coordinate_converter import CoordinateConverter
from util.visualizer import Visualizer
from util.settings_loader import SettingsLoader
from util.evaluation import Evaluation

import glob
import time
import argparse
import os
from pathlib import Path

# Tracking systems
from tracker.SORT.sort import Sort
from tracker.SORT.kalmanBoxTracker import KalmanBoxTracker
from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker
from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker

# Detection systems
from detector.yolo.yolo_ultralytics import YoloUltralyticsDetector
from detector.detr.detr_huggingface import DetrHuggingFaceDetector
from detector.maskfrcnn.maskfrcnn_detectron2 import MaskFasterRCNNDetectron2Detector
from detector.frcnn.frcnn_detectron2 import FasterRCNNDetectron2Detector
from detector.pointnet.pointnet_trainer import PointnetTrainer

from datasets.kitti3D import Kitti3D, CLASSES
from inference_engine import InferenceEngine
from util.logging_config import LoggingConfig
from entities.detection import convert_to_tensor

if __name__ == "__main__":
    logging_config = LoggingConfig()
    logger = logging_config.get_logger(__name__)

    settings = SettingsLoader.load("settings.yaml")
    if settings.runtime.mode == "inference":
        inference_engine = InferenceEngine(settings)
        dataset = inference_engine.load()
        # Object Detection
        detections = inference_engine.predict(detector_name=settings.runtime.detector,
                                 dataset_path=settings.paths.dataset_path,
                                 detection_path=settings.paths.detection_path,
                                 model_path=settings.paths.model_root)
        inference_engine.evaluate_detection(detections=detections,
                                            classes=CLASSES) 
        # Object Tracking
        if settings.runtime.tracker != ""
            inference_engine.update_tracker()

    elif settings.runtime.mode == "train":
        train_dataset = Kitti3D(
            data_root=settings.paths.dataset_path,
            split="training",
            mode="object",
            num_points=1024,
            include_background=True,
        )

        val_dataset = Kitti3D(
            data_root=settings.paths.dataset_path,
            split="val",
            mode="object",
            num_points=1024,
            include_background=True,
        )
        trainer = PointnetTrainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_checkpoint=settings.paths.model_path,
            epochs=20,
            batch_size=16,
            num_points=1024,
            learning_rate=1e-3,
            use_intensity=True,
        )
        trainer.train()
