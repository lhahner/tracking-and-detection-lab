from detector.detector import Detector
from mmdet3d.apis import init_model, inference_detector

class PointpillarsDetector(Detector):
    def __init__(self, input_path, output_path, config_path, checkpoint_path):
        self.input_path = input_path
        self.output_path = output_path
        self.config_file = config_path # './model/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
        self.checkpoint_file = model_path #'./model/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
        self.model = init_model(self.config_file, self.checkpoint_file)
        
    def detect(self):
        """
        TODO
        Detect objects using DETR model loaded.
        
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
            detection_results, data = inference_detector(model, frame)
            
            for detection_result in detection_results:
                self.detections.append(self.format_detections(frame_index, detection_result))
                self.write_output(self.format_detections(frame_index, detection_result)) 
            frame_index += 1
        return self.detections
    
    def format_detections(self, frame_index, results):
        """
        TODO
        Format the detections from Yolo in the required format so
        that the tracking system is able to handle it.
        We require the following string: <frame-id>,-1,x,y,w,h,confidence,-1,-1,-1
        """
        
        return lines
    
    def write_output(self, lines):
        """
        TODO
        Append multiple lines to the output file.
        `self.output_path` should be a FILE path, e.g. ".../det.txt"
        """
    
    def read_data(self):
        """
        TODO
        Read data (frames) from file-system and return them
        as Iteralable object where I can access each frame in detect.
        """
       
    def get_model(self):
        return self.model 
        