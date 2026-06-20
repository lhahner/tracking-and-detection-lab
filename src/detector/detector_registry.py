import importlib
import yaml
from typing import Any, Callable
from dataclasses import dataclass
from definitions import ROOT_DIR
from pathlib import Path
import os


class Registry:
    def __init__(self) -> None:
        self.detector_modules = {} 
        self._classes: dict[str, type] = {}
        self._loaded_modules: set[str] = set()

    def register(self, *names: str) -> Callable[[type], type] | type:
        """Register a class under its class name and optional aliases."""
        if len(names) == 1 and isinstance(names[0], type):
            return self._register_class(names[0])

        def decorator(cls: type) -> type:
            return self._register_class(cls, *names)

        return decorator

    def _register_class(self, cls: type, *aliases: str) -> type:
        for name in (cls.__name__, *aliases):
            self._register_name(name, cls)
        return cls

    def _register_name(self, name: str, cls: type) -> None:
        existing = self._classes.get(name)
        if existing is not None and not self._is_same_class(existing, cls):
            raise KeyError(f"Class {name!r} is already registered")
        self._classes[name] = cls

    def _is_same_class(self, existing: type, cls: type) -> bool:
        return (
            existing is cls
            or (
                existing.__module__ == cls.__module__
                and existing.__name__ == cls.__name__
            )
        )

    def names(self) -> list[str]:
        """Return registered class names."""
        return list(self._classes.keys())

    def get_class(self, name: str) -> type:
        """Return the class object itself."""
        self.__ensure_loaded(name)
        if name not in self._classes:
            raise KeyError(
                f"Unknown class {name!r}. Available: {self.names()}"
            )
        return self._classes[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Create a new instance of the registered class."""
        cls = self.get_class(name)
        return cls(*args, **kwargs)

    def __ensure_loaded(self, name: str) -> None:
        detector_modules = self.__load_from_config()
        module_name = self.detector_modules.get(name)
        if module_name is None or module_name in self._loaded_modules:
            return
        importlib.import_module(module_name)
        self._loaded_modules.add(module_name)

    def __load_from_config(self):
        config_path = Path(os.path.join(ROOT_DIR, "src/detector/detector.yaml")).resolve()
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            self.detector_modules = data.get("detectors", {})


MODELS = Registry()
