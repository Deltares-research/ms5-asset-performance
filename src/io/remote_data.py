"""
Generic I/O utilities for remote data access.

Reads/writes data from a remote folder specified in a .env file.
Each case study has its own .env with a REMOTE_PATH variable pointing
to the shared data folder.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Union
from dotenv import load_dotenv


def get_remote_path(env_path: Union[Path, str] = None) -> Path:
    """Load remote data folder path from a .env file.

    Args:
        env_path: Path to the .env file. If None, searches for .env
            in the current working directory.

    Returns:
        Path to remote data folder.

    Raises:
        ValueError: If REMOTE_PATH not set in the .env file.
    """
    if env_path is None:
        env_path = Path.cwd() / ".env"
    load_dotenv(env_path)

    remote_path = os.environ.get("REMOTE_PATH")
    if remote_path is None:
        raise ValueError(f"REMOTE_PATH not set in {env_path}")

    return Path(remote_path)


def load_json(
    filename: str,
    remote_path: Union[Path, str] = None,
    subfolder: str = "input",
) -> Any:
    """Load a JSON file from the remote folder.

    Args:
        filename: Name of the JSON file.
        remote_path: Root remote path. If None, calls get_remote_path().
        subfolder: Subfolder within remote_path (default "input").

    Returns:
        Parsed JSON content.
    """
    if remote_path is None:
        remote_path = get_remote_path()
    filepath = Path(remote_path) / subfolder / filename
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(
    data: Any,
    filename: str,
    remote_path: Union[Path, str] = None,
    subfolder: str = "output/results",
) -> None:
    """Save data as a JSON file in the remote folder.

    Args:
        data: Data to save (must be JSON-serializable).
        filename: Name of the JSON file.
        remote_path: Root remote path. If None, calls get_remote_path().
        subfolder: Subfolder within remote_path (default "output/results").
    """
    if remote_path is None:
        remote_path = get_remote_path()
    filepath = Path(remote_path) / subfolder / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
