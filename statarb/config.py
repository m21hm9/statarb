import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(dotenv_path: str | None = None) -> None:
    """
    Load environment variables from a .env file if present.

    By default this looks for a .env file in the project root.
    """
    if dotenv_path is None:
        # Assume this file lives in statarb/config.py
        project_root = Path(__file__).resolve().parents[1]
        dotenv_path = project_root / ".env"

    if Path(dotenv_path).exists():
        load_dotenv(dotenv_path)


def get_env(name: str, default: str | None = None) -> str | None:
    """
    Small wrapper to read environment variables.
    """
    return os.getenv(name, default)


def get_alpaca_credentials() -> dict[str, str]:
    """
    Central place to access Alpaca-related configuration.
    """
    load_env()

    return {
        "api_key": get_env("ALPACA_API_KEY", ""),
        "api_secret": get_env("ALPACA_API_SECRET", ""),
        "data_base_url": get_env("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"),
    }

