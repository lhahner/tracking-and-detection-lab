from typing import MappingView
import git
import os
import tomli
import sys 
import subprocess
import argparse

from git.exc import GitCommandError
from importlib.metadata import entry_points
from registry import MODELS, TRACKER, PRE_PROCESSING, POST_PROCESSING, DATASETS, Registry
from definitions import PLUGIN_DIR, ROOT_DIR
from pathlib import Path
from plugin_loader.models.plugin_types import PluginType

MODULES_DIR = os.path.join(ROOT_DIR, "src")
GROUPS = [[MODELS, os.path.join(MODULES_DIR, "detector", "detector.yaml")],
          [TRACKER, os.path.join(MODULES_DIR, "tracker", "tracker.yaml")],
          [DATASETS, os.path.join(MODULES_DIR, "datasets", "dataset.yaml")],
          [PRE_PROCESSING, os.path.join(MODULES_DIR, "pre_processing", "pre_processing.yaml")],
          [POST_PROCESSING, os.path.join(MODULES_DIR, "post_processing", "post_processing.yaml")]]


class PluginInstaller:
    def load_plugin(self, path, config_path, registry, group, name):
        # How to make this safe?
        subprocess.run(
            [sys.executable, "-m", "pip", "install", path],
            shell=False,
            check=True,
            timeout=120,
        ) 
        plugins = entry_points(group=group, name=name)
        for entry_point in plugins:
            clazz = entry_point.load()

            registry.register(
                clazz
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
        group, entries = next(iter(mainfest["project"]["entry-points"].items()))
        impl_name, target = next(iter(entries.items()))

        plugin_type = PluginType[mainfest["tool"]["type"].upper()]
        registry_meta = GROUPS[plugin_type.value]
        self.load_plugin(path=os.path.join(PLUGIN_DIR, name),
                         config_path=registry_meta[1],
                         registry=registry_meta[0],
                         group=group,
                         name=impl_name)


