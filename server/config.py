from pathlib import Path

SERVER_VERSION = "1.1.0-field"

DATA_DIR = Path("./mgsa_data")
DATA_DIR.mkdir(exist_ok=True)

TRACK_DIR = DATA_DIR / "tracks"
TRACK_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "mgsa.db"
