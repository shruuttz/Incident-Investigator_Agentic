
"""Environment configuration for the Manufacturing Investigator agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ("GEMINI_API_KEY",)

# Gemini model configuration
SPECIALIST_MODEL = os.environ.get(
    "SPECIALIST_MODEL", "gemini-2.5-flash"
)
SUPERVISOR_MODEL = os.environ.get(
    "SUPERVISOR_MODEL", "gemini-2.5-flash"
)

SPECIALIST_MAX_TURNS = int(
    os.environ.get("SPECIALIST_MAX_TURNS", "8")
)
SUPERVISOR_MAX_TURNS = int(
    os.environ.get("SUPERVISOR_MAX_TURNS", "8")
)


def require_api_key() -> None:
    """Warn when the Gemini API key is missing."""
    missing = [
        name for name in REQUIRED_ENV_VARS
        if not os.environ.get(name)
    ]

    if missing:
        print(
            "[investigator] Warning: Missing environment variable(s): "
            + ", ".join(missing)
        )
