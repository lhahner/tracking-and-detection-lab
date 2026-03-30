# core/settings.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class PathsConfig:
    """Store resolved filesystem paths used by the application."""

    mot_root: Path
    output_root: Path
    detections_root: Path
    models_root: Path
    detection_path: str
    ground_truth_path: str | None
    tracking_path: str | None

@dataclass(frozen=True)
class RuntimeConfig:
    """Store runtime settings that control execution behavior."""

    display: bool
    dataset: str
    detector: str
    tracker: str
    datatype: str
    benchmark: bool

@dataclass(frozen=True)
class Settings:
    """Represent the fully parsed application configuration."""

    project_name: str
    seed: int
    paths: PathsConfig
    runtime: RuntimeConfig
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
        if not settings.paths.mot_root.exists():
            raise ValueError(f"mot_root not found: {settings.paths.mot_root}")
