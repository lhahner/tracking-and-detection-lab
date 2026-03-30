import pytest
from detector.YOLO.yolo import YoloDetector
import os

@pytest.fixture
def detector():
    """Create a YOLO detector instance for integration-style tests."""
    return YoloDetector(
        "/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/data/MOT15/train/KITTI-17/img1/",
        "/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/src/data/yolo/KITTI-17/det/det.txt",
        "./detector/model/yolo11n.pt",
    )

def test_read_data(detector):
    """Verify that frame paths can be read from the configured dataset."""
    sorted_frames_concat = detector.read_data()
    # assert sorted_frames_concat[0][0] == "/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/data/MOT15/train/KITTI-17/img1/000006.jpg"

def test_format_detections(detector):
    """Verify that raw model predictions can be converted to MOT format."""
    detection_result = detector.get_model()("/home/lennart/Dokumente/gau/advanced-research-training-applied-system-development/track-lab/data/MOT15/train/KITTI-17/img1/000001.jpg")
    format_det = detector.format_detections(detector.read_data()[0], detection_result[0]) 
    assert format_det != None

def test_detect(detector):
    """Verify that the detector produces a non-empty detection result object."""
    detections = detector.detect()
    assert detections != None 
