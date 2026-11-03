"""
System 2 — DEBATE-LITE: Two interpreters + analyst + clarifier. No Devil's Advocate or Judge.
"""
from oracle.agents import interpreter_a, interpreter_b, ambiguity_analyst, clarifier
from oracle.config import AMBIGUITY_THRESHOLD


def run_debate_lite(query: str) -> dict:
    interp_a = interpreter_a(query)
    interp_b = interpreter_b(query)
    analyst = ambiguity_analyst(query, interp_a, interp_b)

    if analyst.ambiguity_score <= AMBIGUITY_THRESHOLD:
        return {
            "is_ambiguous": False,
            "analyst": analyst,
            "final_question": None,
        }

    question = clarifier(query, analyst)
    return {
        "is_ambiguous": True,
        "interpretation_a": interp_a,
        "interpretation_b": interp_b,
        "analyst": analyst,
        "final_question": question,
    }
