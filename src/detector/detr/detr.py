from transformers import AutoImageProcessor, DetrForObjectDetection
from detector.detector import Detector
import os
import torch
from PIL import Image

class DetrDetector(Detector):
    def __init__(self, input_path, output_path, threshold=0.9):
        self.input_path = input_path
        self.output_path = output_path
        self.threshold = threshold
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.detections = []
        
    def detect(self):
        """
        Detect objects using DETR model loaded.
        
        Returns a map of detections, where the index is the frame and the
        value is the corresponding bounding-box and confidence score.
        """
        concat_frames, _ = self.read_data()
        frame_index = 1
        output_file = os.path.join(self.output_path, "det.txt")
        with open(output_file, 'r') as file_obj:
            first_char = file_obj.read(1)
            if first_char:
                print(f"file {output_file} not empty, don't rewrite")
                for line in file_obj:
                    self.detections.append(line)
                return self.detections
            else:
                print(f"file empty running detection on given dataset.")
        
        for concat_frame in concat_frames:
            image = Image.open(concat_frame).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            detection_results = self.image_processor.post_process_object_detection(
                outputs=outputs, threshold=self.threshold, target_sizes=target_sizes
            )[0]
            lines = self.format_detections(frame_index, detection_results)
            self.detections.append(lines)
            self.write_output(lines)
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
        
        xyxy = results["boxes"].detach().cpu().numpy()  # (N,4) x1,y1,x2,y2
        conf = results["scores"].detach().cpu().numpy()  # (N,)
        
        lines = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
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

        with open(os.path.join(self.output_path, "det.txt"), "a", encoding="utf-8") as f:
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
