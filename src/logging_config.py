import os
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent
LOG_PATH = _SCRIPT_PATH / "logs" / "app.log"

os.makedirs(LOG_PATH.parent, exist_ok=True)
