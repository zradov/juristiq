import os
import logging
from pathlib import Path
from datetime import datetime
from .common import OUTPUT_DIR


LOG_PATH = OUTPUT_DIR / "logs" / f"app_{datetime.now().strftime('%H%M%S_%d%m%y')}.log"
os.makedirs(LOG_PATH.parent, exist_ok=True)


def configure_logging() -> None:
    """
    Configures file and console loggers and the logging format.
    """
    file_log_handler = logging.FileHandler(filename=LOG_PATH, encoding="utf-8")
    file_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(handlers=[file_log_handler,  console_log_handler], level=logging.INFO)
