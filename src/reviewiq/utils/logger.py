"""Logging setup for ReviewIQ.

Call ``setup_logging()`` once at the entry point of any script or server.
After that, all modules can use ``logging.getLogger(__name__)`` normally.

Usage
-----
    from reviewiq.utils.logger import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
    logger.info("ReviewIQ started.")
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
from pathlib import Path

import yaml

from reviewiq.utils.config import _find_repo_root

_LOGGING_CONFIG = _find_repo_root() / "configs" / "logging.yaml"


def setup_logging(config_path: str | Path | None = None) -> None:
    """Configure logging from the YAML logging config file.

    Parameters
    ----------
    config_path:
        Path to a logging YAML file. Defaults to ``configs/logging.yaml``.
    """
    path = Path(config_path) if config_path else _LOGGING_CONFIG

    # Ensure log directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    if path.exists():
        with open(path) as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Fallback: basic console logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
