from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

class LoggingConfig:
      def __init__(self, log_dir="logs", log_filename="app.log", log_level=logging.INFO):
          self.log_dir = Path(log_dir)
          self.log_dir.mkdir(parents=True, exist_ok=True)

          self.root_logger = logging.getLogger()

          self._log_filename = log_filename
          self._log_level = _resolve_level(log_level) 
          self._log_format = (
              "%(asctime)s | %(levelname)-8s | %(name)s | "
              "%(filename)s:%(lineno)d | %(message)s"
          )

          self.formatter = logging.Formatter(self._log_format)
          self.file_handler = self._create_file_handler()
          self.console_handler = logging.StreamHandler()

          self._configure_handlers()
          self._configure_root_logger()

      def _resolve_level(self, level):
          if isinstance(level, str):
              return getattr(logging, level.upper(), logging.INFO)
          return level

      def _create_file_handler(self):
          return RotatingFileHandler(
              self.log_dir / self._log_filename,
              maxBytes=5_000_000,
              backupCount=5,
              encoding="utf-8",
          )

      def _configure_handlers(self):
          self.file_handler.setLevel(self._log_level)
          self.file_handler.setFormatter(self.formatter)

          self.console_handler.setLevel(self._log_level)
          self.console_handler.setFormatter(self.formatter)

      def _configure_root_logger(self):
          self.root_logger.handlers.clear()
          self.root_logger.setLevel(self._log_level)
          self.root_logger.addHandler(self.file_handler)
          self.root_logger.addHandler(self.console_handler)

      @property
      def log_level(self):
          return self._log_level

      @log_level.setter
      def log_level(self, value):
          self._log_level = self._resolve_level(value)
          self._configure_handlers()
          self.root_logger.setLevel(self._log_level)

      @property
      def log_filename(self):
          return self._log_filename

      @log_filename.setter
      def log_filename(self, value):
          self._log_filename = value

          self.root_logger.removeHandler(self.file_handler)
          self.file_handler.close()

          self.file_handler = self._create_file_handler()
          self._configure_handlers()
          self.root_logger.addHandler(self.file_handler)

      @property
      def log_format(self):
          return self._log_format

      @log_format.setter
      def log_format(self, value):
          self._log_format = value
          self.formatter = logging.Formatter(self._log_format)
          self._configure_handlers()

      def get_logger(self, name):
          return logging.getLogger(name)