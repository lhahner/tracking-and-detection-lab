from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import yaml


THIRD_PARTY_SIMPLETRACK_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "SimpleTrack"
if str(THIRD_PARTY_SIMPLETRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_SIMPLETRACK_ROOT))

from mot_3d.data_protos import BBox
from mot_3d.frame_data import FrameData
from mot_3d.mot import MOTModel


class SimpleTrack:
    """Project wrapper around the vendored upstream SimpleTrack tracker.

    The wrapper accepts frame-wise 3D LiDAR detections, forwards them to the
    upstream ``MOTModel``, and persists a lightweight MOT-style BEV projection
    for the project's existing tracking evaluation helper.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        default_config = THIRD_PARTY_SIMPLETRACK_ROOT / "configs" / "nu_configs" / "giou.yaml"
        default_output = Path("src/tracker/simpleTrack/tracks/nuScenes_tracks.txt")

        self.config_path = Path(config_path) if config_path is not None else default_config
        self.output_path = Path(output_path) if output_path is not None else default_output
        self._tracker = MOTModel(self._load_config(self.config_path))

    def track(self, detections) -> list[dict]:
        """Track a frame sequence using upstream SimpleTrack.

        Args:
            detections: Sequence of frame dictionaries. Each frame contains
                3D detector output in ``bbox_3d`` format:
                ``[x, y, z, yaw, length, width, height, score]``.

        Returns:
            list[dict]: One item per frame containing normalized track output.
        """
        frame_entries = self._normalize_frames(detections)
        tracking_results: list[dict] = []
        mot_rows: list[list[float]] = []

        for frame_entry in frame_entries:
            frame_number = int(frame_entry["frame"])
            frame_data = self._build_frame_data(frame_entry)
            frame_tracks = self._tracker.frame_mot(frame_data)

            normalized_tracks = []
            for bbox, track_id, state, det_type in frame_tracks:
                public_track_id = int(track_id) + 1
                bbox_array = BBox.bbox2array(bbox).astype(float).tolist()
                normalized_tracks.append(
                    {
                        "track_id": public_track_id,
                        "bbox_3d": bbox_array,
                        "state": state,
                        "label": det_type,
                    }
                )
                mot_rows.append(self._bbox_to_mot_row(frame_number, public_track_id, bbox))

            tracking_results.append({"frame": frame_number, "tracks": normalized_tracks})

        self._write_tracks(mot_rows)
        return tracking_results

    def _load_config(self, config_path: Path) -> dict:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _normalize_frames(self, detections) -> list[dict]:
        if isinstance(detections, dict):
            frames = detections.get("frames", [detections])
        else:
            frames = list(detections)

        normalized = []
        for frame_entry in frames:
            if "frame" not in frame_entry or "detections" not in frame_entry:
                raise ValueError(
                    "Each frame must provide 'frame' and 'detections' keys for SimpleTrack.track()."
                )
            normalized.append(frame_entry)
        return normalized

    def _build_frame_data(self, frame_entry: dict) -> FrameData:
        det_arrays = [self._normalize_detection(det) for det in frame_entry.get("detections", [])]
        det_types = [det.get("label", "object") for det in frame_entry.get("detections", [])]
        aux_info = dict(frame_entry.get("aux_info", {}))
        aux_info.setdefault("is_key_frame", True)

        ego = np.asarray(frame_entry.get("ego", np.eye(4)), dtype=float)
        time_stamp = float(frame_entry.get("time_stamp", frame_entry["frame"]))
        point_cloud = frame_entry.get("pc")
        if point_cloud is not None:
            point_cloud = np.asarray(point_cloud, dtype=float)

        return FrameData(
            dets=det_arrays,
            ego=ego,
            time_stamp=time_stamp,
            pc=point_cloud,
            det_types=det_types,
            aux_info=aux_info,
        )

    def _normalize_detection(self, detection: dict) -> np.ndarray:
        bbox = detection.get("bbox_3d")
        if bbox is None:
            raise ValueError("Each detection must provide 'bbox_3d' for SimpleTrack integration.")

        bbox = np.asarray(bbox, dtype=float)
        if bbox.shape[0] == 7:
            score = float(detection.get("score", 1.0))
            bbox = np.concatenate([bbox, np.asarray([score], dtype=float)])
        elif bbox.shape[0] >= 8:
            bbox = bbox[:8]
        else:
            raise ValueError("bbox_3d must have 7 or 8 values: [x,y,z,yaw,l,w,h,(score)]")
        return bbox

    def _bbox_to_mot_row(self, frame_number: int, track_id: int, bbox: BBox) -> list[float]:
        x = float(bbox.x - bbox.l / 2.0)
        y = float(bbox.y - bbox.w / 2.0)
        w = float(bbox.l)
        h = float(bbox.w)
        score = float(bbox.s if bbox.s is not None else 1.0)
        return [frame_number, track_id, x, y, w, h, score, -1, -1, -1]

    def _write_tracks(self, rows: Iterable[list[float]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    f"{int(row[0])},{int(row[1])},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f},{int(row[7])},{int(row[8])},{int(row[9])}\n"
                )
