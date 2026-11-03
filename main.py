"""
Quick smoke-test: run all 3 systems on a single query.
"""
import json
from oracle.pipeline import run_oracle
from oracle.debate_lite import run_debate_lite
from oracle.baseline import run_baseline_gpt, run_baseline_claude


def pretty(label: str, data) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data if isinstance(data, dict) else data.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    query = input("Enter a query: ").strip()

    pretty("SYSTEM 1 — BASELINE (GPT-4o)", run_baseline_gpt(query))
    pretty("SYSTEM 1 — BASELINE (Claude)", run_baseline_claude(query))
    pretty("SYSTEM 2 — DEBATE-LITE", run_debate_lite(query))
    pretty("SYSTEM 3 — ORACLE (full pipeline)", run_oracle(query))
