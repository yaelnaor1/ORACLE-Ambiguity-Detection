"""
Verbose test — shows exactly what each of the 6 agents said, step by step.
"""
from data.dataset_loader import load_csv
from oracle.agents import (
    interpreter_a, interpreter_b,
    ambiguity_analyst, clarifier,
    devils_advocate, judge,
)
from oracle.config import AMBIGUITY_THRESHOLD
from oracle.models import ClarifierResult

SEP = "-" * 60


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_verbose(query: str, ground_truth: bool, gold_question: str):
    print(f"\nQUERY        : {query}")
    print(f"GROUND TRUTH : ambiguous={ground_truth}")
    print(f"GOLD QUESTION: {gold_question or '(none — clear query)'}")

    # ── ROUND 1: Independent Interpretation ──────────────────
    section("ROUND 1 — INTERPRETER A  (GPT-4o)")
    interp_a = interpreter_a(query)
    print(f"  Answer          : {interp_a.answer}")
    print(f"  Assumed context : {interp_a.assumed_context}")
    print(f"  Confidence      : {interp_a.confidence}")

    section("ROUND 1 — INTERPRETER B  (Claude)")
    interp_b = interpreter_b(query)
    print(f"  Answer          : {interp_b.answer}")
    print(f"  Assumed context : {interp_b.assumed_context}")
    print(f"  Confidence      : {interp_b.confidence}")

    # ── ROUND 2: Ambiguity Analyst ────────────────────────────
    section("ROUND 2 — AMBIGUITY ANALYST  (GPT-4o)")
    analyst = ambiguity_analyst(query, interp_a, interp_b)
    print(f"  Conflict detected : {analyst.conflict_detected}")
    print(f"  Ambiguity type    : {analyst.ambiguity_type}")
    print(f"  Ambiguity score   : {analyst.ambiguity_score}  (threshold={AMBIGUITY_THRESHOLD})")
    print(f"  Reasoning         : {analyst.reasoning}")

    if analyst.ambiguity_score <= AMBIGUITY_THRESHOLD:
        section("RESULT — QUERY IS CLEAR (below threshold)")
        print(f"  System says : NOT ambiguous")
        print(f"  Ground truth: ambiguous={ground_truth}")
        print(f"  Verdict     : {'CORRECT' if not ground_truth else 'FALSE NEGATIVE'}")
        return

    # ── ROUND 3: Clarifier + Devil's Advocate ─────────────────
    section("ROUND 3 — CLARIFIER v1  (Claude)")
    q_v1 = clarifier(query, analyst)
    print(f"  Question v1 : {q_v1}")

    section("ROUND 3 — DEVIL'S ADVOCATE  (GPT-4o)")
    advocate = devils_advocate(query, q_v1, analyst)
    print(f"  Critique              : {advocate.critique}")
    print(f"  Suggested improvement : {advocate.suggested_improvement}")

    section("ROUND 3 — CLARIFIER v2  (Claude, after critique)")
    q_v2 = clarifier(query, analyst, critique=advocate.critique)
    print(f"  Question v2 : {q_v2}")

    # ── ROUND 4: Judge ────────────────────────────────────────
    section("ROUND 4 — JUDGE  (Claude)")
    transcript = (
        f"Clarifier v1: {q_v1}\n"
        f"Devil's Advocate critique: {advocate.critique}\n"
        f"Suggested improvement: {advocate.suggested_improvement}\n"
        f"Clarifier v2: {q_v2}"
    )
    verdict = judge(query, q_v1, q_v2, transcript)
    print(f"  v1 scores  — Relevance: {verdict.relevance_v1}  Specificity: {verdict.specificity_v1}  Necessity: {verdict.necessity_v1}")
    print(f"  v2 scores  — Relevance: {verdict.relevance_v2}  Specificity: {verdict.specificity_v2}  Necessity: {verdict.necessity_v2}")
    print(f"  Winner     : {verdict.winner.upper()}")
    print(f"  Explanation: {verdict.explanation}")

    final = q_v1 if verdict.winner == "v1" else q_v2
    section("FINAL OUTPUT")
    print(f"  System question : {final}")
    print(f"  Gold question   : {gold_question}")
    print(f"  Ground truth    : ambiguous={ground_truth}")
    print(f"  Verdict         : {'CORRECT (true positive)' if ground_truth else 'FALSE POSITIVE'}")


if __name__ == "__main__":
    import sys
    dataset = load_csv()

    # Usage:
    #   python -m evaluation.test_verbose 1          → first ambiguous entry
    #   python -m evaluation.test_verbose 2          → first clear entry
    #   python -m evaluation.test_verbose "my query" → custom query
    choice = sys.argv[1] if len(sys.argv) > 1 else "1"

    if choice == "1":
        row = next(r for r in dataset if r["is_ambiguous"])
        run_verbose(row["query"], row["is_ambiguous"], row["gold_question"])

    elif choice == "2":
        row = next(r for r in dataset if not r["is_ambiguous"])
        run_verbose(row["query"], row["is_ambiguous"], row["gold_question"])

    else:
        run_verbose(choice, ground_truth=None, gold_question="")
