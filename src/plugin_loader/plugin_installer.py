from typing import MappingView
import git
import os
import tomli

from git.exc import GitCommandError
from importlib.metadata import entry_points
from src.registry import MODELS, TRACKER, PRE_PROCESSING, POST_PROCESSING, DATASETS, Registry
from definitions import PLUGIN_DIR, ROOT_DIR
from pathlib import Path

MODULES_DIR = os.path.join(ROOT_DIR, "src")
GROUPS = {"tracking-and-detection-lab.detector": [MODELS, os.path.join(MODULES_DIR,
                                                                       "detector",
                                                                       "detector.yaml")],
          "tracking-and-detection-lab.tracker": [TRACKER, os.path.join(MODULES_DIR,
                                                                       "tracker",
                                                                       "tracker.yaml")],
          "tracking-and-detection-lab.datasets": [DATASETS, os.path.join(MODULES_DIR,
                                                                         "datasets",
                                                                         "dataset.yaml")],
          "tracking-and-detection-lab.pre-processing": [PRE_PROCESSING, os.path.join(MODULES_DIR,
                                                                                     "pre_processing",
                                                                                     "pre_processing.yaml")],
          "tracking-and-detection-lab.post-processing": [POST_PROCESSING, os.path.join(MODULES_DIR,
                                                                                       "post_processing",
                                                                                       "post_processing.yaml")]}


class PluginInstaller:
    def load_plugin(self, config_path, registry, group):
        plugins = entry_points(group=group)
        registry = Registry(config_path)
        for entry_point in plugins:
            model_class = entry_point.load()

            registry.register(
                entry_point.name,
                model_class
            )

    def load_all_plugins(self):
        for group in GROUPS.keys():
            self.load_plugin(config_path=GROUPS.get(group)[1],
                             registry=GROUPS.get(group)[0],
                             group=group)

    def install_plugin(self, name, source, git=False):
        if git:
            try:
                git.Git(PLUGIN_DIR).clone(source)
            except GitCommandError:
                raise ModuleNotFoundError("Git Repository seems not to exists or is not public.\
                                           If you dont want to make it public, clone the repo manually,\
                                           and install plugin by providing local path while running\
                                           installation.")
        if not os.path.exists(os.path.join(PLUGIN_DIR, name)):
            raise FileNotFoundError("The plugin path does not exists, \
                                     validate that ./plugins dir exists.")
        mainfest_path = os.path.join(PLUGIN_DIR, name, "pyproject.toml")
        if not os.path.exists(mainfest_path):
            raise FileNotFoundError("Mainfest config not found, create a pyproject.toml")

        with Path(mainfest_path).open("rb") as f:
            mainfest = tomli.load(f)
        plugin_type = mainfest["type"]
        group = mainfest["project"]["entry-points"]["tracking-and-detection-lab"]
        registry_meta = GROUPS.get(group + "." + plugin_type)
        self.load_plugin(config_path=registry_meta[1],
                         registry=registry_meta[0],
                         group=group)
