import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from helpers.helpers import create_dummy_plugin, clean_up_dummy_folder
from plugin_loader.plugin_installer import PluginInstaller
from src.registry import MODELS

class TestPluginIntegration(unittest.TestCase):
    def test_dummy_plugin(self):
        path = create_dummy_plugin()
        PluginInstaller().install_plugin(name="dummy_plugin",
                                         source=path,
                                         git=False)
        self.assertTrue(len(MODELS > 0))
        clean_up_dummy_folder(path)
