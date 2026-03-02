from detector.detector import Detector
from ultralytics import YOLO
import os

class YoloDetector(Detector):
    def __init__(self, input_path, output_path, model_path):
        self.input_path = input_path
        self.output_path = output_path
        self.model = YOLO(model_path)
        self.detections = []
     
    def detect(self):
        """
        Detect objects using YOLO model loaded.
        
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
            detection_results = self.model(concat_frame)
            for detection_result in detection_results:
                self.detections.append(self.format_detections(frame_index, detection_result))
                self.write_output(self.format_detections(frame_index, detection_result)) 
            frame_index += 1
        return self.detections
        
    def format_detections(self, frame_index, results):
        """
        Format the detections from Yolo in the required format so
        that the tracking system is able to handle it.
        We require the following string: <frame-id>,-1,x,y,w,h,confidence,-1,-1,-1
        """
        if results is None:
            raise ValueError("The given results object is None.")
        if frame_index is None:
            raise ValueError("The given frame object is None")
        
        xyxy = results.boxes.xyxy.cpu().numpy()   # (N,4) -> x1,y1,x2,y2 in original image space
        conf = results.boxes.conf.cpu().numpy()   # (N,)
        
        lines = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            # MOT format expects top-left x,y plus width,height.
            w = x2 - x1
            h = y2 - y1
            line = f"{frame_index},-1,{x1:.0f},{y1:.0f},{w:.3f},{h:.3f},{c:.6f},-1,-1,-1\n"
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
        sorted_frames_concat = [str(self.input_path) + frame for frame in frames]
        return sorted_frames_concat, frames
       
    def get_model(self):
        return self.model
        
           
       
