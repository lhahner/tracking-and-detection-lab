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

class TestLoggerInt(unittest.TestCase):
    def test_basic_logging(self):
        logging_config = LoggingConfig()
        logger = logging_config.get_logger(__name__)
        
        logger.info("test logging")
        path = os.path.join(PROJECT_ROOT, 'logs/app.log')
        print(path) 
        self.assertTrue(
            os.path.exists(path)
        )
        self.assertTrue(
            os.stat(path).st_size != 0
        )
        # clean up 
        os.remove(path)

         