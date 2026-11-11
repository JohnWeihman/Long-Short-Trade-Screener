"""Project settings: paths and values read from the environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_UNIVERSE = PROJECT_ROOT / "config" / "universe.csv"

_PLACEHOLDER_MARKERS = ("your name", "your.email", "example.com")


class ConfigError(RuntimeError):
    """A required setting is missing or still a placeholder."""


def data_dir() -> Path:
    """Root of the local data cache (override with SCREENER_DATA_DIR)."""
    override = (os.environ.get("SCREENER_DATA_DIR") or "").strip()
    return Path(override) if override else PROJECT_ROOT / "data"


def sec_user_agent() -> str:
    """The User-Agent the SEC requires: a real name and contact email."""
    value = (os.environ.get("SEC_USER_AGENT") or "").strip()
    if not value:
        raise ConfigError(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and put your "
            "name and email in it."
        )
    if "@" not in value or any(m in value.lower() for m in _PLACEHOLDER_MARKERS):
        raise ConfigError(
            "SEC_USER_AGENT still looks like the placeholder. Put your real "
            "name and email in .env."
        )
    return value
