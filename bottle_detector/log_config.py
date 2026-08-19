from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .paths import resolve_app_path

LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "bottle_detector.log"
_configured = False


def configure_logging(*, level: int = logging.INFO) -> None:
    """Wires the root `bottle_detector` logger to both the console and a
    rotating log file. Console output alone isn't enough for the packaged
    GUI/EXE, where nobody is watching a terminal on the actual production
    line -- the log file is what makes a failed or oddly-behaving run
    diagnosable after the fact."""
    global _configured
    if _configured:
        return
    _configured = True

    log_dir = resolve_app_path(LOG_DIR_NAME)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("bottle_detector")
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_dir / LOG_FILE_NAME, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
