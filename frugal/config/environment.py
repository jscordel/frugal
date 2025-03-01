from os import environ
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(
    environ.get("PROJECT_DIR")
)

DATA_DIR = Path(
    environ.get(
        "DATA_DIR",
        PROJECT_DIR / "data"
    )
)
