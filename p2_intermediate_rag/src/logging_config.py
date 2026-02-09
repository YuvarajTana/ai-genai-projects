"""
Logging Configuration
=====================
Centralized logger setup using loguru.
"""

from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure loguru logger with level, file output, and format.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None = no file output)
        format_string: Log format string (None = default)
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Add stderr handler
    import sys
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
        )
