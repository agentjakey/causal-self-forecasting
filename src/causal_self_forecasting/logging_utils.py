"""Structured logging.

Human-readable on the console, JSON lines to a file when a run directory exists. The JSON
log is part of a run's evidence, so it records the same fields every time rather than
free-form messages.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOGGER_NAME = "csf"
_configured = False


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload["fields"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=True)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<7} {record.getMessage()}"
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict) and extra:
            pairs = " ".join(f"{key}={value}" for key, value in extra.items())
            base = f"{base}  [{pairs}]"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Set up the `csf` logger. Safe to call more than once."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    if not _configured:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(ConsoleFormatter())
        logger.addHandler(console)
        _configured = True

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        already_attached = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == path.resolve()
            for handler in logger.handlers
        )
        if not already_attached:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(JsonLineFormatter())
            logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def log_event(level: int, message: str, **fields: Any) -> None:
    """Log a message with structured fields attached."""
    get_logger().log(level, message, extra={"fields": fields})


def info(message: str, **fields: Any) -> None:
    log_event(logging.INFO, message, **fields)


def warn(message: str, **fields: Any) -> None:
    log_event(logging.WARNING, message, **fields)


def error(message: str, **fields: Any) -> None:
    log_event(logging.ERROR, message, **fields)


def debug(message: str, **fields: Any) -> None:
    log_event(logging.DEBUG, message, **fields)
