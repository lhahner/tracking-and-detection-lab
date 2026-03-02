import numpy as np

class CoordinateConverter:
    def __init__(self):
        pass
    
    def convert2DDetectionToBox(self, seq_dets, frame):
        if len(seq_dets) < 1:
            raise ValueError("The sequence detections are empty")
    
        dets = seq_dets[seq_dets[:,0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]
        return dets
