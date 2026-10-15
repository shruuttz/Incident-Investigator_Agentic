
"""Shared asynchronous Gemini runner for investigator agents."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types


_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Create the Gemini client once and reuse it."""
    global _client

    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in your local environment or deployment settings."
            )
        _client = genai.Client(api_key=api_key)

    return _client


async def generate_text(
    *,
    model: str,
    system_instruction: str,
    prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Generate a response, optionally constrained to a JSON schema."""
    client = get_client()

    config_args: dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": 0.1,
    }

    if response_schema is not None:
        config_args["response_mime_type"] = "application/json"
        config_args["response_schema"] = response_schema

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_args),
    )

    result = response.text
    if not result:
        raise RuntimeError("Gemini returned an empty response.")

    return result


async def generate_json(
    *,
    model: str,
    system_instruction: str,
    prompt: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    """Generate and parse a structured JSON response."""
    result = await generate_text(
        model=model,
        system_instruction=system_instruction,
        prompt=prompt,
        response_schema=response_schema,
    )

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Gemini returned JSON, but the result was not an object."
        )

    return parsed
