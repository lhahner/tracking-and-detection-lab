import importlib
import yaml
import os 

from typing import Any
from definitions import ROOT_DIR
from pathlib import Path
from config.logging_config import logged

@logged
class Registry:

    def __init__(self, config_path) -> None:
        self.config_path = config_path
        self.modules = {}
        self._classes: dict[str, type] = {}
        self.__loaded_modules: set[str] = set()
        
    def register_all(self, type):
        config_path = Path(os.path.join(ROOT_DIR, self.config_path)).resolve()
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if data is None or data[type] is None:
                self.log.warning(f"The given type {type}, has no modules to register.")
                return ""
            for key in data[type].keys():
                self.__ensure_loaded(name=key, type=type)

    def register(self, *names: str):
        """
        There are two possible ways to register a class
        to the registry. Example 1, put the decorator
        at the top of the class;
        >>> @MODEL.register
        >>> class X ...

        Class X will be registered in _classes. Example 2;
        put an alias instead;
        >>> @MODEL.register("X_2")
        >>> class X

        This causes the class to be registered as the alias.
        """
        if len(names) == 1 and isinstance(names[0], type):
            return self.__register_class(names[0])

        def decorator(cls: type) -> type:
            return self.__register_class(cls, *names)

        return decorator

    def __register_class(self, cls: type, *aliases: str) -> type:
        for name in (cls.__name__, *aliases):
            self.__register_name(name, cls)
        return cls

    def __register_name(self, name: str, cls: type) -> None:
        existing = self._classes.get(name)
        if existing is not None and not self.__is_same_class(existing, cls):
            raise KeyError(f"Class {name!r} is already registered")
        self._classes[name] = cls

    def __is_same_class(self, existing: type, cls: type) -> bool:
        return (
            existing is cls
            or (
                existing.__module__ == cls.__module__
                and existing.__name__ == cls.__name__
            )
        )

    def names(self) -> list[str]:
        """
        Return all registered class names
        as strings.
        """
        return list(self._classes.keys())

    def get_class(self, name: str, type) -> type:
        """
        Return the class object itself by
        getting the class or alias string
        name.
        """
        self.__ensure_loaded(name, type)
        if name not in self._classes:
            raise KeyError(
                f"Unknown class {name!r}. Available: {self.names()}"
            )
        return self._classes[name]

    def create(self, name: str, type, *args: Any, **kwargs: Any) -> Any:
        """
        Create the model class instance with the given set of
        of parameters. Every model needs to have the same set
        of parameters though.
        """
        cls = self.get_class(name, type)
        return cls(*args, **kwargs)

    def __ensure_loaded(self, name: str, type) -> None:
        modules = self.__load_from_config(type)
        module_name = self.modules.get(name)
        if module_name is None or module_name in self.__loaded_modules:
            return
        module = importlib.import_module(module_name)
        self.__loaded_modules.add(module_name)
        self._classes[module_name] = module.__name__

    def __load_from_config(self, type):
        config_path = Path(os.path.join(ROOT_DIR, self.config_path)).resolve()
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            self.modules = data.get(type, {})


DATASETS = Registry("src/datasets/dataset.yaml")
TRACKER = Registry("src/tracker/tracker.yaml")
MODELS = Registry("src/detector/detector.yaml")
POST_PROCESSING = Registry("src/post_processing/post_processing.yaml")
PRE_PROCESSING = Registry("src/pre_processing/pre_processing.yaml")
