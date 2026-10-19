"""
Task 1 - Configure logging.

Creates one shared logger ('ml-service') that writes to:
  * the console (so you can watch it live in the VS Code terminal)
  * a rotating file, logs/service.log (max 1 MB, keeps 3 backups)
Every line has a timestamp, the level and the message.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "service.log")


def get_logger(name: str = "ml-service", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call many times."""
    logger = logging.getLogger(name)
    logger.setLevel(level)  # INFO for production, DEBUG while developing

    # Guard: without this, every re-import (e.g. Flask reloader) adds
    # another pair of handlers and each line gets printed twice.
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    fileh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3,
                                encoding="utf-8")
    fileh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fileh)
    logger.propagate = False  # don't also send lines to the root logger
    return logger


# Allow the level to be switched without editing code: LOG_LEVEL=DEBUG
logger = get_logger(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(),
                                  logging.INFO))
