from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from .file_utils import LOGS_DIR, OUTPUTS_DIR, ensure_dir


LOGGER_NAME = "fraudshield"


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    target_dir = ensure_dir(log_dir or LOGS_DIR)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for filename, level in [("startup.log", logging.INFO), ("error.log", logging.ERROR), ("fraudshield.log", logging.INFO)]:
        file_handler = logging.FileHandler(target_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_logger() -> logging.Logger:
    return setup_logging()


def log_exception(message: str, exc: BaseException) -> None:
    get_logger().exception("%s: %s", message, exc)


def install_exception_hook() -> None:
    def _handle_exception(exc_type, exc_value, exc_traceback) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        get_logger().error("Unhandled exception\n%s", text.rstrip())
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _handle_exception
