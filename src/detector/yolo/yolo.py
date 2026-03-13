from detector.detector import Detector
from ultralytics import YOLO
import os
from pathlib import Path

class YoloDetector(Detector):
    def __init__(self, input_path, output_path, model_path):
        self.input_path = input_path
        self.output_path = output_path
        self.output_file = Path(output_path) / "det.txt"
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
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.touch(exist_ok=True)
        with open(self.output_file, "r", encoding="utf-8") as file_obj:
            first_char = file_obj.read(1)
            if first_char:
                print(f"file {self.output_file} not empty, don't rewrite")
                for line in file_obj:
                    self.detections.append(line)
                return self.detections
            else:
                print(f"file empty running detection on given dataset.")
        
        for frame, concat_frame in zip(frames, concat_frames):
            detection_results = self.model(concat_frame)
            for detection_result in detection_results:
                formatted = self.format_detections(frame_index, detection_result)
                self.detections.append(formatted)
                self.write_output(formatted)
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
        cls = results.boxes.cls.cpu().numpy()     # (N,) COCO class ids, person=0
        
        lines = []
        for (x1, y1, x2, y2), c, class_id in zip(xyxy, conf, cls):
            if int(class_id) != 0:
                continue
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
        if not self.output_file.parent.exists():
            raise ValueError(f"Output directory does not exist: {self.output_file.parent}")

        with open(self.output_file, "a", encoding="utf-8") as f:
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
        sorted_frames_concat = [os.path.join(str(self.input_path), frame) for frame in frames]
        return sorted_frames_concat, frames
       
    def get_model(self):
        return self.model
        
           
       
