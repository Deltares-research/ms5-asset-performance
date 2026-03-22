"""
I/O utilities for the D-Sheet piling case study.

Reads/writes data from a remote folder specified in .env.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Remote path configuration
# -----------------------------------------------------------------------------

def get_remote_path() -> Path:
    """
    Load remote data folder path from environment file.

    Returns:
        Path to remote data folder.

    Raises:
        ValueError: If REMOTE_DATA_PATH not set in environment.
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    remote_path = os.environ.get("REMOTE_PATH")
    if remote_path is None:
        raise ValueError("REMOTE_PATH not set in .env")

    return Path(remote_path)


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load JSON file from remote folder.

    Args:
        filename: Name of JSON file (relative to remote folder).

    Returns:
        Parsed JSON content.
    """
    filepath = get_remote_path() / f"input/{filename}"
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save JSON file to remote folder.

    Args:
        data: Data to save.
        filename: Name of JSON file (relative to remote folder).
    """
    filepath = get_remote_path() / f"output/results/{filename}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    pass

