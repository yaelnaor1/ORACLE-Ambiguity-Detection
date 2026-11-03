"""
Generates human-readable "reasoning files" — one Markdown file per query —
capturing the full decision trail behind ORACLE's (and Debate-Lite's) ambiguity
classification and clarifying-question generation: each interpreter's read on
the query, the analyst's diagnosis, the devil's-advocate critique, and the
judge's verdict.

This is the artifact for "understanding the decisions": a transcript of *why*
the system asked what it asked (or chose not to ask), not just *what* it asked.
None of this reasoning is stored in results.csv, so generating it means
re-running the pipeline — each query costs ~7 LLM calls for ORACLE plus ~3 for
Debate-Lite. Keep --n small unless you specifically need the full set.

Usage:
  python -m evaluation.reasoning_files            # 5 queries (default — cheap preview)
  python -m evaluation.reasoning_files --n 20     # first 20 (10 ambiguous + 10 clear)
"""
import argparse
import csv
from pathlib import Path

from oracle.pipeline import run_oracle
from oracle.debate_lite import run_debate_lite

DATASET_PATH = Path("data/ambig_qa.csv")
OUT_DIR      = Path("evaluation/reasoning_files")


def _load_queries(n: int) -> list[dict]:
    """Pick an even mix of ambiguous + clear queries."""
    rows = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))
    half = n // 2
    ambiguous = [r for r in rows if r["is_ambiguous"] == "True"][:half]
    clear     = [r for r in rows if r["is_ambiguous"] == "False"][:n - half]
    return ambiguous + clear


def _fmt_interp(label: str, interp) -> str:
    return (
        f"**{label}** (confidence {interp.confidence:.2f})\n"
        f"- Answer: {interp.answer}\n"
        f"- Assumed context: {interp.assumed_context}\n"
    )


def _format_oracle(ora) -> str:
    lines = ["## ORACLE — full 6-agent pipeline", ""]

    lines.append("### Round 1 — Independent interpretations")
    lines.append(_fmt_interp("Interpreter A", ora.interpretation_a))
    lines.append(_fmt_interp("Interpreter B", ora.interpretation_b))

    lines.append("### Round 2 — Analyst diagnosis")
    lines.append(f"- Ambiguity type: **{ora.analyst.ambiguity_type}**")
    lines.append(f"- Ambiguity score: **{ora.analyst.ambiguity_score:.2f}**")
    lines.append(f"- Conflict detected: {ora.analyst.conflict_detected}")
    lines.append(f"- Reasoning: {ora.analyst.reasoning}")
    lines.append("")

    if not ora.is_ambiguous:
        lines.append("**Verdict: query treated as CLEAR — answered directly, no clarification round.**")
        return "\n".join(lines)

    lines.append("### Round 3 — Clarification + challenge")
    lines.append(f"- Clarifier v1: {ora.clarifier.question_v1}")
    lines.append(f"- Devil's Advocate critique: {ora.advocate.critique}")
    lines.append(f"- Suggested improvement: {ora.advocate.suggested_improvement}")
    lines.append(f"- Clarifier v2 (revised): {ora.clarifier.question_v2}")
    lines.append("")

    lines.append("### Round 4 — Judge's verdict")
    j = ora.judge
    lines.append(f"- v1 scores — relevance {j.relevance_v1:.2f}, specificity {j.specificity_v1:.2f}, necessity {j.necessity_v1:.2f}")
    lines.append(f"- v2 scores — relevance {j.relevance_v2:.2f}, specificity {j.specificity_v2:.2f}, necessity {j.necessity_v2:.2f}")
    lines.append(f"- **Winner: {j.winner}**")
    lines.append(f"- Explanation: {j.explanation}")
    lines.append("")
    lines.append(f"### Final question asked\n> {ora.final_question}")

    return "\n".join(lines)


def _format_debate_lite(dl: dict) -> str:
    lines = ["", "## Debate-Lite — 2 interpreters + analyst + clarifier (no challenge round)", ""]
    if not dl["is_ambiguous"]:
        lines.append("**Verdict: query treated as CLEAR — answered directly.**")
        return "\n".join(lines)

    a = dl["analyst"]
    lines.append(f"- Ambiguity type: **{a.ambiguity_type}** (score {a.ambiguity_score:.2f})")
    lines.append(f"- Reasoning: {a.reasoning}")
    lines.append(f"- Final question: {dl['final_question']}")
    return "\n".join(lines)


def generate(n: int):
    queries = _load_queries(n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in OUT_DIR.glob("*.md")}

    total = len(queries)
    print(f"Generating reasoning files for {total} queries "
          f"(~{total * 10} LLM calls total — ORACLE + Debate-Lite)\n")

    for i, row in enumerate(queries, 1):
        qid = row["id"]
        if qid in existing:
            print(f"[{i}/{total}] skip {qid} (already generated)")
            continue

        query = row["query"]
        safe_q = query.encode("ascii", errors="replace").decode("ascii")
        print(f"[{i}/{total}] {safe_q[:70]}...")

        ora = run_oracle(query)
        dl  = run_debate_lite(query)

        text = (
            f"# {query}\n\n"
            f"Ground truth: {'AMBIGUOUS' if row['is_ambiguous'] == 'True' else 'CLEAR'}"
            + (f" — gold question: {row['gold_question']}" if row.get("gold_question") else "")
            + "\n\n"
            + _format_oracle(ora) + "\n"
            + _format_debate_lite(dl) + "\n"
        )
        (OUT_DIR / f"{qid}.md").write_text(text, encoding="utf-8")

    print(f"\nDone. Reasoning files saved to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5,
                        help="Number of queries to generate reasoning files for (default: 5)")
    args = parser.parse_args()
    generate(args.n)
