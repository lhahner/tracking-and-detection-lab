from pathlib import Path
from copy import deepcopy

import torch
try:
    from mmengine.dataset import Compose, pseudo_collate
    from mmdet3d.apis import init_model
    from mmdet3d.structures import get_box_type
except (ImportError, ModuleNotFoundError) as exc:
    raise ImportError("FCOS3DMMDetections3D requires MMDetection3D. Install the OpenMMLab stack first.") from exc
from detector.detector import Detector
from entities.detection import Detection, DetectionSequence, FrameDetection
from config.logging_config import LoggingConfig
from definitions import ROOT_DIR

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = (
    Path(ROOT_DIR)
    / "third_party"
    / "mmdetection3d"
    / "configs"
    / "fcos3d"
    / "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py"
)
DEFAULT_CHECKPOINT_FILE = (
    PROJECT_DIR
    / "model"
    / "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"
)
logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class FCOS3DMMDetections3D(Detector):
    def __init__(self,
                 dataset,
                 classes,
                 settings,
                 config_file=DEFAULT_CONFIG_FILE,
                 checkpoint_file=DEFAULT_CHECKPOINT_FILE,
                 batch_size=16):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.classes = classes
        self.batch_size = batch_size
        self.settings = settings
        self.model = init_model(
            str(self.config_file),
            str(self.checkpoint_file)
        )
        dataset_cfg = self.model.cfg.test_dataloader.dataset
        self.test_pipeline = Compose(deepcopy(dataset_cfg.pipeline))
        self.box_type_3d, self.box_mode_3d = get_box_type(dataset_cfg.box_type_3d)
        self.class_map = self.__build_class_map(self.classes)

    def detect(self):
        """Run MMDetection3D inference and return project detection objects."""
        detection_sequence = DetectionSequence()
        for item in self.dataset:
            instance_data = self.__predict_instances(item)
            detections, highest_score_index = self.__convert_instances(instance_data)
            detection_sequence.frames.append(FrameDetection(frame=item["sample_id"],
                                                            highest_score_index=highest_score_index,
                                                            dets=detections,
                                                            targets=item["target"],))
        return detection_sequence

    def __build_class_map(self, classes):
        if not isinstance(classes, dict):
            return torch.as_tensor(classes, dtype=torch.long)

        model_classes = None
        if hasattr(self.model, "dataset_meta"):
            model_classes = self.model.dataset_meta.get("classes")
        if model_classes is None and hasattr(self.model, "cfg"):
            model_classes = self.model.cfg.get("class_names", None)
        if model_classes is None:
            raise ValueError("Unable to determine MMDetection3D SSN class order")

        try:
            return torch.tensor([classes[name] for name in model_classes], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"MMDetection3D class {exc.args[0]!r} is not present in the project label map") from exc

    def __predict_instances(self, item):
        data = self.__prepare_mono_input(item)
        with torch.no_grad():
            result = self.model.test_step(pseudo_collate([data]))[0]
        if hasattr(result, "pred_instances_3d"):
            return result.pred_instances_3d
        raise ValueError("MMDetection3D inference result does not contain pred_instances_3d")

    def __prepare_mono_input(self, item):
        camera_channel = getattr(self.dataset, "camera_channel", "CAM_FRONT")
        camera = item["calib"]["cameras"][camera_channel]
        data = {
            "images": {
                camera_channel: {
                    "img_path": item["images"][camera_channel],
                    "cam2img": camera["camera_intrinsic"],
                }
            },
            "box_type_3d": self.box_type_3d,
            "box_mode_3d": self.box_mode_3d,
        }
        return self.test_pipeline(data)

    def __convert_instances(self, instance_data):
        scores = instance_data.scores_3d.detach().cpu().float()
        bboxes_3d = instance_data.bboxes_3d
        bbox_tensor = bboxes_3d.tensor if hasattr(bboxes_3d, "tensor") else bboxes_3d
        bboxes = bbox_tensor.detach().cpu().float()
        labels_reference = instance_data.labels_3d.detach().cpu().long()
        if bboxes.numel() == 0:
            return [], None
        if bboxes.ndim != 2 or bboxes.shape[1] < 7:
            raise ValueError(f"Expected MMDetection3D boxes with shape [N, >=7], got {tuple(bboxes.shape)}")
        bboxes = bboxes[:, :7].clone()
        bboxes[:, 2] += bboxes[:, 5] / 2

        labels = self.__map_labels(labels_reference)
        highest_score_index = int(scores.argmax().item())
        return [
            Detection(score=float(score.item()), label=int(label.item()), box=box)
            for score, box, label in zip(scores, bboxes, labels)
        ], highest_score_index

    def __map_labels(self, labels_reference):
        if self.class_map.numel() == 0:
            return labels_reference
        valid = (labels_reference >= 0) & (labels_reference < self.class_map.numel())
        if not torch.all(valid):
            raise ValueError("MMDetection3D FCOS3D returned a class index outside"
                             "the configured label map")
        return self.class_map[labels_reference]
