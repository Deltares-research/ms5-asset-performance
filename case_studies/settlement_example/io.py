"""
I/O utilities for the settlement case study.

Thin wrapper around src.io that binds this case study's .env path.
"""

import json
from pathlib import Path
from typing import Dict, Any
from src.io import get_remote_path as _get_remote_path, save_json as _save_json

_ENV_PATH = Path(__file__).parent / ".env"


def get_remote_path() -> Path:
    """Load remote path from this case study's .env file."""
    return _get_remote_path(env_path=_ENV_PATH)


def load_json(filename: str) -> Any:
    """Load JSON file from this case study's remote input folder."""
    filepath = get_remote_path() / f"input/{filename}"
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: Any, filename: str) -> None:
    """Save JSON file to this case study's remote output folder."""
    _save_json(data, filename, remote_path=get_remote_path())
