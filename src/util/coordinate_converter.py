import numpy as np

class CoordinateConverter:
    def __init__(self):
        pass
    
    def convert3DLIDARDetectionToBEV(self, seq_dets, frame):
        if len(seq_dets) < 1:
            raise ValueError("The sequence detections are empty")
        else:
            dets = seq_dets[seq_dets[:, 0] == frame, 2:8]
            x = dets[:, 0]
            y = dets[:, 1]
            w = dets[:, 3] 
            l = dets[:, 4]
            x1 = x - l / 2.0
            x2 = x + l / 2.0
            y1 = y - w / 2.0
            y2 = y + w / 2.0

            # Map from LiDAR meters to BEV pixel coordinates
            x_range = (0.0, 70.4)
            y_range = (-40.0, 40.0)
            resolution = 0.1
            W = int((x_range[1] - x_range[0]) / resolution)
            H = int((y_range[1] - y_range[0]) / resolution)

            x1 = (x1 - x_range[0]) / resolution
            x2 = (x2 - x_range[0]) / resolution
            y1 = (y1 - y_range[0]) / resolution
            y2 = (y2 - y_range[0]) / resolution

            x1 = np.clip(x1, 0, W - 1)
            x2 = np.clip(x2, 0, W - 1)
            y1 = np.clip(y1, 0, H - 1)
            y2 = np.clip(y2, 0, H - 1)

            score = np.ones_like(x1)
            dets = np.stack([x1, y1, x2, y2, score], axis=1)
        return dets

    
    def convert2DDetectionToBox(self, seq_dets, frame):
        if len(seq_dets) < 1:
            raise ValueError("The sequence detections are empty")
    
        dets = seq_dets[seq_dets[:,0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]
        return dets
