"""Logging configuration utilities."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    console: Console | None = None,
) -> None:
    """
    Set up logging configuration.

    Parameters
    ----------
    level : int
        Logging level
    format_string : str | None
        Custom format string
    console : Console | None
        Rich console instance
    """
    if console is None:
        console = Console()

    # Create rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )

    # Set format
    if format_string is None:
        format_string = "%(message)s"

    rich_handler.setFormatter(logging.Formatter(format_string))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add rich handler
    root_logger.addHandler(rich_handler)

    # Configure specific loggers
    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("bitsandbytes").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
