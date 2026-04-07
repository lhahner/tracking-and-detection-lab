from transformers import AutoImageProcessor, DetrForObjectDetection
from detector.detector import Detector
import os
import torch
from PIL import Image

class DetrHuggingFaceDetector(Detector):
    """Run DETR-based person detection and export MOT-format detections."""

    def __init__(self, input_path, output_path, threshold=0.9):
        """Initialize the DETR detector.

        Args:
            input_path: Directory containing input image frames.
            output_path: Directory where `det.txt` will be written.
            threshold: Confidence threshold used during post-processing.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.threshold = threshold
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.person_label_ids = {
            label_id
            for label_id, label_name in self.model.config.id2label.items()
            if label_name.lower() == "person"
        }
        self.detections = []
        
    def detect(self):
        """Run DETR inference on every frame and persist MOT detections.

        Returns:
            list[list[str]] | list[str]: Collected detection lines, either read
            from an existing output file or generated during inference.
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
        """Convert DETR detections into MOT challenge text lines.

        Args:
            frame_index: One-based frame index.
            results: DETR post-processed prediction dictionary.

        Returns:
            list[str]: MOT-format detection lines for person detections only.

        Raises:
            ValueError: If `results` or `frame_index` is missing.
        """
        if results is None:
            raise ValueError("The given results object is None.")
        if frame_index is None:
            raise ValueError("The given frame object is None")
        
        xyxy = results["boxes"].detach().cpu().numpy()  # (N,4) x1,y1,x2,y2
        conf = results["scores"].detach().cpu().numpy()  # (N,)
        labels = results["labels"].detach().cpu().numpy()  # (N,)
        
        lines = []
        for (x1, y1, x2, y2), c, label_id in zip(xyxy, conf, labels):
            if int(label_id) not in self.person_label_ids:
                continue
            w = x2 - x1
            h = y2 - y1
            line = f"{frame_index},-1,{x1:.0f},{y1:.0f},{w:.3f},{h:.3f},{c:.6f},-1,-1,-1\n"
            lines.append(line)
        
        return lines
    
    def write_output(self, lines):
        """Append detection lines to the detector output file.

        Args:
            lines: MOT-format lines to append.

        Raises:
            ValueError: If the output directory does not exist.
        """
        out_dir = os.path.dirname(self.output_path) or "."
        if not os.path.exists(out_dir):
            raise ValueError(f"Output directory does not exist: {out_dir}")

        with open(os.path.join(self.output_path, "det.txt"), "a", encoding="utf-8") as f:
            f.writelines(lines)
    
    def read_data(self):
        """Read and sort image frame paths from the input directory.

        Returns:
            tuple[list[str], list[str]]: Absolute frame paths and corresponding
            frame filenames.

        Raises:
            ValueError: If the input directory does not exist.
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
        """Return the loaded DETR model instance."""
        return self.model 
