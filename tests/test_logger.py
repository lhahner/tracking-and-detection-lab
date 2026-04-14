import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from util.logging_config import LoggingConfig

class TestLogger(unittest.TestCase):
    @patch("util.logging_config.Path.mkdir")
    @patch("util.logging_config.RotatingFileHandler")
    def test_resolve_level_from_log_level_enum(self, mock_file_handler_cls, mock_mkdir):
        mock_file_handler = MagicMock()
        mock_file_handler_cls.return_value = mock_file_handler

        logging_config = LoggingConfig(log_dir="logs")
        resolved = logging_config._resolve_level("INFO")
        self.assertEqual(resolved, logging.INFO)

