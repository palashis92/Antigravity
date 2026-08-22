"""Structured logging subsystem for LUMI with robust UTF-8 encoding support."""

from __future__ import annotations

import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Ensure standard output uses UTF-8 on Windows consoles to support Bangla Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


class ColoredFormatter(logging.Formatter):
    """Terminal formatter with ANSI color codes for readability during development."""

    GREY = "\033[38;20m"
    BLUE = "\033[34;20m"
    CYAN = "\033[36;20m"
    GREEN = "\033[32;20m"
    YELLOW = "\033[33;20m"
    RED = "\033[31;20m"
    BOLD_RED = "\033[31;1m"
    RESET = "\033[0m"

    FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"

    FORMATS = {
        logging.DEBUG: CYAN + FORMAT + RESET,
        logging.INFO: GREEN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


_ROOT_INITIALIZED = False


def setup_logger(
    name: str = "lumi",
    log_level: str = "INFO",
    log_file: Optional[str | Path] = "logs/lumi.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return a structured logger with UTF-8 stream handling."""
    global _ROOT_INITIALIZED

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _ROOT_INITIALIZED:
        # Console handler with UTF-8 stream wrapper
        console_stream = sys.stdout
        if hasattr(console_stream, "buffer"):
            console_stream = io.TextIOWrapper(
                console_stream.buffer, encoding="utf-8", errors="backslashreplace", line_buffering=True
            )

        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

        # Rotating file handler (UTF-8 encoded)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

        _ROOT_INITIALIZED = True

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger for a specific module."""
    return logging.getLogger(f"lumi.{name}")
