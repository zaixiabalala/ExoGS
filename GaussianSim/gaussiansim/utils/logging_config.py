import sys
import logging
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None, log_format=None):
    """
    Setup logging configuration with console and optional file output.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
        log_format: Optional custom log format string
    
    Returns:
        Configured root logger
    """
    if log_format is None:
        log_format = '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
    
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger
