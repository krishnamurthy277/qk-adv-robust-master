# Adapted for this project; original logic reviewed and re-validated.
"""Structured logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: Path = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger with console and optional file output.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
