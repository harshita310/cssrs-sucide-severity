"""
Logging utilities for notebooks and scripts.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False


def get_logger(name: str = "mentalbert_cssr", level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger with a consistent console format.

    Safe to call repeatedly; handlers are attached only once to the root
    project logger to avoid duplicate notebook output.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        root = logging.getLogger("mentalbert_cssr")
        if not root.handlers:
            root.addHandler(handler)
            root.propagate = False
        _CONFIGURED = True

    return logger
