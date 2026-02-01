from detector.detector import Detector
from mmdet3d.apis import init_model, inference_detector
import os

class PointpillarsDetector(Detector):
    def __init__(self, input_path, output_path, config_path, checkpoint_path):
        self.input_path = input_path
        self.output_path = output_path
        self.config_file = config_path # './model/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        self.checkpoint_file = checkpoint_path #'./model/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        self.model = init_model(self.config_file, self.checkpoint_file, device='cpu')
        self.detections = []
        
    def detect(self):
        """
        Detect objects using PointPillars model loaded.
        
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
            detection_results, data = inference_detector(self.model, concat_frame)
            
            for detection_result in detection_results.pred_instances_3d:
                self.detections.append(self.format_detections(frame_index, detection_result))
                self.write_output(self.format_detections(frame_index, detection_result)) 
            frame_index += 1
        return self.detections
    
    def format_detections(self, frame_index, results):
        """
        TODO
        Format the detections from PointPillars in the required format so
        that the tracking system is able to handle it.
        We require the following string: <frame-id>,-1,x,y,z,w,l,r confidence,-1,-1,-1
        - where x,y,z represent the center of the box
        - and w is the width
        - and l is the length
        - and r is the rotation / yaw angle
        """
        if results is None:
            raise ValueError("The given results object is None.")
        if frame_index is None:
            raise ValueError("The given frame object is None")
        
        xyzlwhr = results.bboxes_3d
        conf = results.scores_3d
        
        lines = []
        for (x, y, z, w, l, h, r), c in zip(xyzlwhr, conf):
            # If you want ints for x,y like your example, round them.
            # Keep w,h and conf as floats.
            line = f"{frame_index},-1,{x:.3f},{y:.3f},{z:.3f},{w:.3f},{l:.3f},{h:.6f},{r:.3f},-1,-1,-1\n"
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

        with open(os.path.join(self.output_path, 'det.txt'), "a", encoding="utf-8") as f:
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
           if file_name.endswith((".bin")):
               frames.append(file_name)
        frames.sort()
        sorted_frames_concat = [self.input_path + frame for frame in frames]
        return sorted_frames_concat, frames
       
    def get_model(self):
        return self.model 
        