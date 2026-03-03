# core/settings.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class PathsConfig:
    mot_root: Path
    output_root: Path
    detections_root: Path
    models_root: Path
    detection_path: str
    ground_truth_path: str | None
    tracking_path: str | None

@dataclass(frozen=True)
class RuntimeConfig:
    display: bool
    dataset: str
    detector: str
    tracker: str
    datatype: str
    benchmark: bool

@dataclass(frozen=True)
class Settings:
    project_name: str
    seed: int
    paths: PathsConfig
    runtime: RuntimeConfig
    raw: dict[str, Any]  

class SettingsLoader:
    @staticmethod
    def load(path):
        cfg_path = Path(path).resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        project = data.get("project", {})
        paths = data.get("paths", {})
        runtime = data.get("runtime", {})

        base = cfg_path.parent
        resolved_paths = PathsConfig(
            mot_root=SettingsLoader.resolve(base, paths.get("mot_root", "../data/MOT15")),
            output_root=SettingsLoader.resolve(base, paths.get("output_root", "./output")),
            detections_root=SettingsLoader.resolve(base, paths.get("detections_root", "./data")),
            models_root=SettingsLoader.resolve(base, paths.get("models_root", "./detector")),
            detection_path=paths.get("detection_path", "data/*/*/det/det.txt"),
            ground_truth_path=paths.get("ground_truth_path"),
            tracking_path=paths.get("tracking_path")
        )

        settings = Settings(
            project_name=project.get("name", "tracking-and-detection-lab"),
            seed=int(project.get("seed", 0)),
            paths=resolved_paths,
            runtime=RuntimeConfig(
                display=bool(runtime.get("display", False)),
                dataset=runtime.get("dataset", "*"),
                detector=runtime.get("detector", "yolo"),
                tracker=runtime.get("tracker", "sort"),
                datatype=runtime.get("datatype"),
                benchmark=bool(runtime.get("benchmark", False))
              ),
            raw=data,
          )
        SettingsLoader.validate(settings)
        return settings

    @staticmethod
    def resolve(base, value):
          if value is None:
              return None
          p = Path(value)
          return (base / p).resolve() if not p.is_absolute() else p

    @staticmethod
    def validate(settings):
        if not settings.paths.mot_root.exists():
            raise ValueError(f"mot_root not found: {settings.paths.mot_root}")
