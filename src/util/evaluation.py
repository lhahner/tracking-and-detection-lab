from pathlib import Path
import numpy as np
import motmetrics as mm

class Evaluation:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.metrics_handler = mm.metrics.create()

    def read_mot_file(self, file_path, filter_ground_truth_by_confidence=False):
        """
        Filters the given ground truth values by frame and 
        returns an object that includes the frame, 
        the object_id and bounding box.
        """
        mot_rows = np.loadtxt(Path(file_path), delimiter=",")
        
        if mot_rows.ndim == 1: 
            mot_rows = mot_rows.reshape(1, -1)
            
        detections_per_frame = {}
        
        for mot_row in mot_rows:
       
            if filter_ground_truth_by_confidence and len(mot_row) > 6 and mot_row[6] <= 0:
                continue
           
            frame_number = int(mot_row[0]); object_id = int(mot_row[1])
            bounding_box_xywh = [mot_row[2], mot_row[3], mot_row[4], mot_row[5]]
            detections_per_frame.setdefault(frame_number, []).append((object_id, bounding_box_xywh))
            
        return detections_per_frame

    def evaluate_sequence(self, ground_truth_file_path, 
                          predicted_tracking_file_path, 
                          sequence_name="sequence", 
                          metrics=None):
        """
        Run the given string of mot-metrics on the prediction compared
        with the given groundtruth, will return a list of scores.
        """
        if metrics is None:
            metrics = ["idf1", "mota", "motp", "precision", "recall"]

        ground_truth_by_frame = self.read_mot_file(ground_truth_file_path, filter_ground_truth_by_confidence=True)
        predicted_tracks_by_frame = self.read_mot_file(predicted_tracking_file_path, filter_ground_truth_by_confidence=False)
        
        mot_accumulator = mm.MOTAccumulator(auto_id=False) # MotMetric setup
        maximum_iou_distance = 1.0 - self.iou_threshold # default to 0.5
        
        for frame_number in sorted(set(ground_truth_by_frame) | set(predicted_tracks_by_frame)):
            ground_truth_items = ground_truth_by_frame.get(frame_number, [])
            predicted_items = predicted_tracks_by_frame.get(frame_number, [])
            
            ground_truth_ids = [item[0] for item in ground_truth_items]
            predicted_ids = [item[0] for item in predicted_items]
            
            ground_truth_boxes_xywh = [item[1] for item in ground_truth_items]
            predicted_boxes_xywh = [item[1] for item in predicted_items]
            
            iou_distance_matrix = mm.distances.iou_matrix(ground_truth_boxes_xywh, 
                                                          predicted_boxes_xywh, 
                                                          max_iou=maximum_iou_distance)
            
            mot_accumulator.update(
                ground_truth_ids, 
                predicted_ids, 
                iou_distance_matrix, 
                frameid=frame_number)
        
        return self.metrics_handler.compute(mot_accumulator, 
                                            metrics, 
                                            name=sequence_name
                                            )
