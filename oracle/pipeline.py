"""
ORACLE pipeline — 4 rounds, 6 agents.
"""
from oracle.agents import (
    interpreter_a,
    interpreter_b,
    ambiguity_analyst,
    clarifier,
    devils_advocate,
    judge,
)
from oracle.config import AMBIGUITY_THRESHOLD
from oracle.models import ClarifierResult, OracleResult


def run_oracle(query: str) -> OracleResult:
    # ── Round 1: Independent Interpretation ──────────────────────────────────
    interp_a = interpreter_a(query)
    interp_b = interpreter_b(query)

    # ── Round 2: Analyst Diagnosis ────────────────────────────────────────────
    analyst = ambiguity_analyst(query, interp_a, interp_b)

    if analyst.ambiguity_score <= AMBIGUITY_THRESHOLD:
        return OracleResult(
            query=query,
            is_ambiguous=False,
            interpretation_a=interp_a,
            interpretation_b=interp_b,
            analyst=analyst,
        )

    # ── Round 3: Clarification + Challenge ────────────────────────────────────
    question_v1 = clarifier(query, analyst)

    advocate = devils_advocate(query, question_v1, analyst)

    question_v2 = clarifier(query, analyst, critique=advocate.critique)

    # ── Round 4: Final Judgment ───────────────────────────────────────────────
    transcript = (
        f"Clarifier v1: {question_v1}\n"
        f"Devil's Advocate critique: {advocate.critique}\n"
        f"Suggested improvement: {advocate.suggested_improvement}\n"
        f"Clarifier v2: {question_v2}"
    )

    verdict = judge(query, question_v1, question_v2, transcript)
    final_question = question_v1 if verdict.winner == "v1" else question_v2

    return OracleResult(
        query=query,
        is_ambiguous=True,
        interpretation_a=interp_a,
        interpretation_b=interp_b,
        analyst=analyst,
        clarifier=ClarifierResult(question_v1=question_v1, question_v2=question_v2),
        advocate=advocate,
        judge=verdict,
        final_question=final_question,
    )
