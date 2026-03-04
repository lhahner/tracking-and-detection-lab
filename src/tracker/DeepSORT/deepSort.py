from __future__ import annotations

import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSortImpl
except ImportError:  # pragma: no cover
    _DeepSortImpl = None


class DeepSort:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.7,
        nn_budget: int | None = 100,
        embedder: str = "mobilenet",
        half: bool = True,
        bgr: bool = True,
        embedder_gpu: bool = True,
        polygon: bool = False,
        today=None,
        gating_only_position: bool = False,
    ) -> None:
        if _DeepSortImpl is None:
            raise ImportError(
                "deep_sort_realtime is not installed. "
                "Install it with: pip install deep-sort-realtime"
            )

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._tracker = _DeepSortImpl(
            max_age=max_age,
            n_init=min_hits,
            max_iou_distance=iou_threshold,
            nn_budget=nn_budget,
            embedder=embedder,
            half=half,
            bgr=bgr,
            embedder_gpu=embedder_gpu,
            polygon=polygon,
            today=today,
            gating_only_position=gating_only_position,
        )

    def _to_deepsort_dets(self, dets: np.ndarray):
        deep_dets = []
        for det in dets:
            x1, y1, x2, y2, score = det[:5]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            deep_dets.append(([float(x1), float(y1), w, h], float(score), "object"))
        return deep_dets

    def update(self, dets=np.empty((0, 5)), frame=None) -> np.ndarray:
        if dets is None:
            dets = np.empty((0, 5), dtype=float)

        dets = np.asarray(dets, dtype=float)
        if dets.size == 0:
            dets = np.empty((0, 5), dtype=float)
        elif dets.ndim != 2 or dets.shape[1] < 5:
            raise ValueError("dets must have shape (N, 5+) with [x1,y1,x2,y2,score].")

        if frame is None:
            raise ValueError(
                "DeepSORT requires the current frame image for appearance embeddings. "
                "Call update(dets, frame=frame_bgr)."
            )

        deep_dets = self._to_deepsort_dets(dets)
        tracks = self._tracker.update_tracks(deep_dets, frame=frame)

        ret = []
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            l, t, r, b = trk.to_ltrb()
            ret.append([float(l), float(t), float(r), float(b), float(trk.track_id)])

        if ret:
            return np.asarray(ret, dtype=float)
        return np.empty((0, 5), dtype=float)
