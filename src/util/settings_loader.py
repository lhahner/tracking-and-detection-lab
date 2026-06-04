# core/settings.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from definitions import ROOT_DIR
import yaml
import os


@dataclass(frozen=True)
class PathsConfig:
    """Store resolved filesystem paths used by the application."""
    dataset_path: Path
    output_root: Path
    detections_root: Path
    models_root: Path
    detection_path: str
    ground_truth_path: str | None
    tracking_path: str | None
    logging_path: str | None
    mmdetection3d_path: str | None

@dataclass(frozen=True)
class BenchmarkConfig:
    """Propreties which define benchmark hyperparameters"""
    iou_threshold: int
    class_filter: list 

@dataclass(frozen=True)
class VisualizerConfig:
    """Define some stuff on how to visualize"""
    colours: str

@dataclass(frozen=True)
class RuntimeConfig:
    """Store runtime settings that control execution behavior."""
    mode: str
    display: bool
    dataset: str
    detector: str
    tracker: str
    datatype: str
    benchmark: bool

@dataclass(frozen=True)
class TrackerConfig:
    max_age: float
    min_hits: float
    iou_threshold: float

@dataclass(frozen=True)
class Settings:
    """Represent the fully parsed application configuration."""
    project_name: str
    seed: int
    paths: PathsConfig
    runtime: RuntimeConfig
    benchmark: BenchmarkConfig
    tracker: TrackerConfig
    visualizer: VisualizerConfig
    raw: dict[str, Any]


class SettingsLoader:
    """Load, resolve, and validate the project's YAML settings file."""

    @staticmethod
    def load(path):
        """Load, resolve, and validate the YAML settings file.

        Args:
            path: Path to the YAML settings file.

        Returns:
            Settings: Parsed and validated application settings.
        """
        cfg_path = Path(os.path.join(ROOT_DIR, path)).resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        project = data.get("project", {})
        paths = data.get("paths", {})
        runtime = data.get("runtime", {})
        benchmark = data.get("benchmark", {})        
        tracker = data.get("tracker", {})        
        visualizer = data.get("visualizer", {})

        base = cfg_path.parent
        resolved_paths = PathsConfig(
            dataset_path=SettingsLoader.resolve(base, paths.get("dataset_path", "../data/")),
            output_root=SettingsLoader.resolve(base, paths.get("output_root", "./output")),
            detections_root=SettingsLoader.resolve(base, paths.get("detections_root", "./data")),
            models_root=SettingsLoader.resolve(base, paths.get("models_root", f"./detector/default-model.pth")),
            detection_path=paths.get("detection_path", "data/*/*/det/det.txt"),
            ground_truth_path=paths.get("ground_truth_path"),
            tracking_path=paths.get("tracking_path"),
            logging_path=paths.get("logging_path"),
            mmdetection3d_path=paths.get("mmdetection3d_path"))
            
        settings = Settings(
            project_name=project.get("name", "tracking-and-detection-lab"),
            seed=int(project.get("seed", 0)),
            paths=resolved_paths,
            runtime=RuntimeConfig(
                mode=runtime.get("mode", "inference"),
                display=bool(runtime.get("display", False)),
                dataset=runtime.get("dataset", "*"),
                detector=runtime.get("detector", "yolo"),
                tracker=runtime.get("tracker", "sort"),
                datatype=runtime.get("datatype"),
                benchmark=bool(runtime.get("benchmark", False))
            ),
            benchmark=BenchmarkConfig(
            iou_threshold=benchmark.get("iou_threshold"),
            class_filter=benchmark.get("class_filter")
            ),
            tracker=TrackerConfig(
                max_age=tracker.get("max_age"),
                min_hits=tracker.get("min_hits"),
                iou_threshold=tracker.get("iou_threshold")
            ),
            visualizer=VisualizerConfig(
                colours=visualizer.get("colour")
            ),
            raw=data,
        )
        return settings

    @staticmethod
    def resolve(base, value):
        """Resolve a path value relative to a base directory when needed.

          Args:
              base: Base directory used for relative paths.
              value: Path-like value to resolve.

          Returns:
              Path | None: Absolute resolved path or `None` when no value is set.
        """
        if value is None:
            return None
        p = Path(value)
        return (base / p).resolve() if not p.is_absolute() else p

    @staticmethod
    def validate(settings):
        """Validate required settings before runtime starts.

        Args:
            settings: Parsed settings object to validate.

        Raises:
            ValueError: If required filesystem paths do not exist.
        """
        if not settings.paths.dataset_path.exists():
            raise ValueError(f"dataset_path not found: {settings.paths.dataset_path}")
