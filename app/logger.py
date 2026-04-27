"""
app/logger.py
─────────────────────────────────────────────────────────────────────────────
Structured logging setup for the Flask API.

Supports two formats configured via .env:
  • json  → machine-readable (production)
  • text  → human-readable (development)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import sys
import time
from typing import Any, Dict

from src.config import LOGGING as LOG_CONFIG


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            log_object.update(record.extra)
        return json.dumps(log_object)


class _TextFormatter(logging.Formatter):
    """Human-readable formatter with colour codes for terminal output."""

    LEVEL_COLOURS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelname, "")
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        msg = record.getMessage()
        base = f"{colour}[{timestamp}] {record.levelname:<8}{self.RESET} | {record.name} | {msg}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def get_logger(name: str = "churn_api") -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers
        return logger

    level = getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if LOG_CONFIG["format"] == "json":
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(_TextFormatter())

    logger.addHandler(handler)

    # File handler
    try:
        file_handler = logging.FileHandler(LOG_CONFIG["log_file"])
        file_handler.setLevel(level)
        file_handler.setFormatter(_JSONFormatter())
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        pass  # Skip file logging if path is unavailable

    return logger
