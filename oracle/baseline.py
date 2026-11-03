"""
System 1 — BASELINE: Single model, direct answer or clarifying question.
"""
from oracle.llm_clients import call_gpt, call_claude

_SYSTEM = """You are a helpful assistant.
If the user's query is ambiguous or unclear, ask ONE clarifying question.
Otherwise answer it directly.
Keep your response concise."""


def run_baseline_gpt(query: str) -> str:
    return call_gpt(_SYSTEM, query)


def run_baseline_claude(query: str) -> str:
    return call_claude(_SYSTEM, query)
