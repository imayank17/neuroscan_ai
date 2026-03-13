import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Ensure log directory exists
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "app.log")

def _setup_logger():
    # Create logger
    logger = logging.getLogger("NeuroScanAPI")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the logger is imported multiple times
    if not logger.handlers:
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | [%(module)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # 2. Rotating File Handler (Max 5MB per file, keep 3 backups)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

# Export a centralized logger instance
app_logger = _setup_logger()
