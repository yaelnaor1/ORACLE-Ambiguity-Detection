"""
Method 2: Automatic Reference-Based Scoring
  - ROUGE-L      : word-overlap between generated question and gold question
  - BERTScore    : semantic similarity between generated question and gold question
  - ARR          : Ambiguity Resolution Rate — boolean, does clarification lead to correct answer?
  - exact_match  : strict normalized match between generated and gold question
  - nearby_match : BERTScore >= 0.85 between generated and gold question
  - answer_match : 3-level answer quality — "exact" | "close" | "far"
"""
from __future__ import annotations
from oracle.llm_clients import call_gpt
from oracle.models import AutoScore


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    h = hypothesis.lower().split()
    r = reference.lower().split()
    lcs = _lcs_length(h, r)
    if lcs == 0:
        return 0.0
    precision = lcs / len(h)
    recall    = lcs / len(r)
    return round(2 * precision * recall / (precision + recall), 4)


# ── BERTScore ─────────────────────────────────────────────────────────────────

_bertscore_model = None

def _get_bertscore():
    global _bertscore_model
    if _bertscore_model is None:
        from bert_score import BERTScorer
        _bertscore_model = BERTScorer(lang="en", rescale_with_baseline=True)
    return _bertscore_model


def bertscore(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    scorer = _get_bertscore()
    _, _, f1 = scorer.score([hypothesis], [reference])
    return round(float(f1[0]), 4)


# ── ARR (Ambiguity Resolution Rate) ──────────────────────────────────────────
# Logic:
#   1. System generates clarifying question Q
#   2. Feed gold answer A as simulated user response
#   3. Ask model to answer the original query given Q + A as context
#   4. Check if final answer contains / matches the gold answer

_ARR_SYSTEM = """You are a helpful assistant.
The user asked a question, then was asked a clarifying question, and provided an answer to it.
Using that context, answer the original question as accurately and concisely as possible."""


def _normalize(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def arr(
    query: str,
    clarifying_question: str,
    gold_answer: str,
) -> bool:
    """Returns True if the model's final answer matches the gold answer."""
    if not clarifying_question or not gold_answer:
        return False

    user_msg = (
        f'Original question: "{query}"\n'
        f'Clarifying question asked: "{clarifying_question}"\n'
        f'User replied: "{gold_answer}"\n\n'
        f"Now answer the original question in one short sentence."
    )
    response = call_gpt(_ARR_SYSTEM, user_msg)
    return _normalize(gold_answer) in _normalize(response)


NEARBY_THRESHOLD = 0.85


def exact_match(hypothesis: str, reference: str) -> bool:
    """Strict normalized string equality between generated and gold question."""
    if not hypothesis or not reference:
        return False
    return _normalize(hypothesis) == _normalize(reference)


def nearby_match(hypothesis: str, reference: str, threshold: float = NEARBY_THRESHOLD) -> bool:
    """Semantic similarity >= threshold between generated and gold question."""
    if not hypothesis or not reference:
        return False
    return bertscore(hypothesis, reference) >= threshold


# ── Multi-reference matching ──────────────────────────────────────────────────
# AmbigQA queries often have several valid gold sub-questions (one per
# disambiguation). Scoring against only the first one unfairly penalizes a
# system that generated a different — but equally valid — gold question.
# These helpers score against the *whole* reference set and keep the best match,
# i.e. cross-validate the system's question against all known-good ground truth.

def split_gold_questions(all_gold_questions: str) -> list[str]:
    """Split the pipe-separated `all_gold_questions` field into a clean list."""
    if not all_gold_questions:
        return []
    return [q.strip() for q in all_gold_questions.split("|") if q.strip()]


def exact_match_multi(hypothesis: str, references: list[str]) -> bool:
    """True if hypothesis exactly matches ANY gold reference question."""
    return any(exact_match(hypothesis, ref) for ref in references)


def nearby_match_multi(hypothesis: str, references: list[str], threshold: float = NEARBY_THRESHOLD) -> bool:
    """True if hypothesis is semantically close to ANY gold reference question."""
    return any(nearby_match(hypothesis, ref, threshold) for ref in references)


def best_bertscore(hypothesis: str, references: list[str]) -> float:
    """Highest BERTScore between hypothesis and any gold reference question."""
    if not hypothesis or not references:
        return 0.0
    return max(bertscore(hypothesis, ref) for ref in references)


def best_rouge_l(hypothesis: str, references: list[str]) -> float:
    """Highest ROUGE-L between hypothesis and any gold reference question."""
    if not hypothesis or not references:
        return 0.0
    return max(rouge_l(hypothesis, ref) for ref in references)


def answer_match_score(
    query: str,
    clarifying_question: str,
    gold_answer: str,
) -> str:
    """
    3-level answer quality after clarification:
      "exact" — normalized final answer matches gold answer
      "close" — BERTScore(final answer, gold answer) >= 0.7
      "far"   — neither
    """
    if not clarifying_question or not gold_answer:
        return "far"

    user_msg = (
        f'Original question: "{query}"\n'
        f'Clarifying question asked: "{clarifying_question}"\n'
        f'User replied: "{gold_answer}"\n\n'
        f"Now answer the original question in one short sentence."
    )
    response = call_gpt(_ARR_SYSTEM, user_msg)

    if _normalize(gold_answer) in _normalize(response):
        return "exact"
    if bertscore(response, gold_answer) >= 0.7:
        return "close"
    return "far"


# ── Combined scorer ───────────────────────────────────────────────────────────

def compute_auto_score(
    generated_question: str,
    gold_question: str,
    gold_answer: str,
    query: str,
    is_ambiguous: bool,
    all_gold_questions: str = "",
) -> AutoScore:
    if not is_ambiguous or not gold_question:
        return AutoScore(
            rouge_l=0.0, bertscore=0.0, arr=None,
            exact_match=None, nearby_match=None, answer_match=None,
        )

    # Cross-validate against every known-good gold question, not just the
    # first one — AmbigQA frequently lists several valid disambiguations.
    references = split_gold_questions(all_gold_questions) or [gold_question]

    rl = best_rouge_l(generated_question, references)
    bs = best_bertscore(generated_question, references)
    ar = arr(query, generated_question, gold_answer) if generated_question and gold_answer else None
    em = exact_match_multi(generated_question, references)
    nm = nearby_match_multi(generated_question, references)
    am = answer_match_score(query, generated_question, gold_answer) if generated_question and gold_answer else None

    return AutoScore(rouge_l=rl, bertscore=bs, arr=ar, exact_match=em, nearby_match=nm, answer_match=am)
