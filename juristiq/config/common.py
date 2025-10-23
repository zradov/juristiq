from pathlib import Path


# Path to the data folder.
DATA_DIR = Path(__file__).parent.parent.parent / "data"
# Folder where scripts output is stored.
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

