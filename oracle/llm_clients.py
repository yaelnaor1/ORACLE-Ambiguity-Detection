import json
import openai
import anthropic
from google import genai
from google.genai import types as genai_types
from oracle.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
    OPENAI_MODEL, ANTHROPIC_MODEL, GEMINI_MODEL, DEEPSEEK_MODEL,
)

import threading as _threading
import httpx as _httpx

_SSL = _httpx.Client(verify=False)
_deepseek_lock = _threading.Lock()

_openai    = openai.OpenAI(api_key=OPENAI_API_KEY, http_client=_SSL)
_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, http_client=_SSL)
_gemini    = genai.Client(api_key=GEMINI_API_KEY)
_deepseek  = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    http_client=_httpx.Client(
        verify=False,
        timeout=_httpx.Timeout(120.0, connect=5.0),
    ),
    max_retries=0,
)


def call_gpt(system: str, user: str) -> str:
    response = _openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def call_claude(system: str, user: str) -> str:
    try:
        response = _anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
    except anthropic.BadRequestError:
        return "[content filtered]"


def call_gemini(system: str, user: str) -> str:
    """Gemini is the independent eval judge — not used in the pipeline itself."""
    prompt = f"{system}\n\n{user}"
    response = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.3),
    )
    return response.text.strip()


def call_deepseek(system: str, user: str) -> str:
    with _deepseek_lock:
        response = _deepseek.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
    return response.choices[0].message.content.strip()


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])
