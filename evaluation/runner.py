"""
Evaluation runner — feeds all 500 queries through all 3 systems.
Applies all 3 scoring methods per row.
Saves row-by-row (safe to resume after a crash).
"""
import csv
import time
import traceback
from pathlib import Path

from data.dataset_loader import load_csv
from oracle.baseline import run_baseline_gpt, run_baseline_claude
from oracle.debate_lite import run_debate_lite
from oracle.pipeline import run_oracle
from oracle.agents import eval_judge
from evaluation.auto_scorer import compute_auto_score

RESULTS_PATH = Path("evaluation/results.csv")

SYSTEMS = ["baseline_gpt", "baseline_claude", "dlite", "oracle"]

def _score_fields(prefix):
    return [
        # Method 1 — Gemini judge (3 axes, kept separate)
        f"{prefix}_relevance", f"{prefix}_specificity", f"{prefix}_necessity",
        f"{prefix}_hallucination", f"{prefix}_false_positive", f"{prefix}_explanation",
        # Method 2 — Automatic
        f"{prefix}_rouge_l", f"{prefix}_bertscore", f"{prefix}_arr",
        f"{prefix}_exact_match", f"{prefix}_nearby_match", f"{prefix}_answer_match",
    ]

FIELDNAMES = (
    ["id", "query", "ground_truth_ambiguous", "gold_question", "gold_answer"] +

    # System 1a: Baseline GPT
    ["baseline_gpt_response"] + _score_fields("baseline_gpt") +

    # System 1b: Baseline Claude
    ["baseline_claude_response"] + _score_fields("baseline_claude") +

    # System 2: Debate-Lite
    ["dlite_is_ambiguous", "dlite_ambiguity_type", "dlite_ambiguity_score",
     "dlite_final_question"] + _score_fields("dlite") +

    # System 3: ORACLE
    ["oracle_is_ambiguous", "oracle_ambiguity_type", "oracle_ambiguity_score",
     "oracle_interp_a_confidence", "oracle_interp_b_confidence",
     "oracle_question_v1", "oracle_question_v2", "oracle_winner",
     "oracle_final_question"] + _score_fields("oracle") +

    ["error"]
)


def _already_done() -> set[str]:
    """Return IDs already processed (including error rows — they will be retried separately)."""
    if not RESULTS_PATH.exists():
        return set()
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f) if not row.get("error")}


def _open_writer():
    is_new = not RESULTS_PATH.exists()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = open(RESULTS_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()
    return f, writer


def _apply_scores(
    result: dict,
    prefix: str,
    query: str,
    ambiguity_type: str,
    clarifying_question: str,
    gold_question: str,
    gold_answer: str,
    is_ambiguous: bool,
    all_gold_questions: str = "",
):
    # Method 1: Gemini judge
    s = eval_judge(query, ambiguity_type, clarifying_question)
    result[f"{prefix}_relevance"]    = s.relevance
    result[f"{prefix}_specificity"]  = s.specificity
    result[f"{prefix}_necessity"]    = s.necessity
    result[f"{prefix}_hallucination"]  = s.hallucination
    result[f"{prefix}_false_positive"] = s.false_positive
    result[f"{prefix}_explanation"]    = s.explanation

    # Method 2: Automatic scores (cross-validated against all gold questions)
    auto = compute_auto_score(clarifying_question, gold_question, gold_answer, query, is_ambiguous,
                              all_gold_questions=all_gold_questions)
    result[f"{prefix}_rouge_l"]      = auto.rouge_l
    result[f"{prefix}_bertscore"]    = auto.bertscore
    result[f"{prefix}_arr"]          = "" if auto.arr is None else auto.arr
    result[f"{prefix}_exact_match"]  = "" if auto.exact_match is None else auto.exact_match
    result[f"{prefix}_nearby_match"] = "" if auto.nearby_match is None else auto.nearby_match
    result[f"{prefix}_answer_match"] = "" if auto.answer_match is None else auto.answer_match


def _run_one(row: dict) -> dict:
    query        = row["query"]
    is_ambiguous = row["is_ambiguous"]
    gold_question = row.get("gold_question", "")
    gold_answer   = row.get("gold_answer", "")
    all_gold_questions = row.get("all_gold_questions", "")

    result = {
        "id":                     row["id"],
        "query":                  query,
        "ground_truth_ambiguous": is_ambiguous,
        "gold_question":          gold_question,
        "gold_answer":            gold_answer,
        "error":                  "",
    }

    # ── Run all 4 systems ────────────────────────────────────────────────────
    gpt_resp  = run_baseline_gpt(query)
    cld_resp  = run_baseline_claude(query)
    dl        = run_debate_lite(query)
    ora       = run_oracle(query)

    # Use ORACLE analyst's ambiguity type for all Gemini judge calls
    # (most informed assessment — runs 2 full interpreters before deciding)
    ambiguity_type = ora.analyst.ambiguity_type

    # ── System 1a: Baseline GPT ──────────────────────────────────────────────
    result["baseline_gpt_response"] = gpt_resp
    _apply_scores(result, "baseline_gpt", query, ambiguity_type,
                  gpt_resp, gold_question, gold_answer, is_ambiguous, all_gold_questions)

    # ── System 1b: Baseline Claude ───────────────────────────────────────────
    result["baseline_claude_response"] = cld_resp
    _apply_scores(result, "baseline_claude", query, ambiguity_type,
                  cld_resp, gold_question, gold_answer, is_ambiguous, all_gold_questions)

    # ── System 2: Debate-Lite ────────────────────────────────────────────────
    dl_question = dl.get("final_question") or ""
    result["dlite_is_ambiguous"]    = dl["is_ambiguous"]
    result["dlite_ambiguity_type"]  = dl["analyst"].ambiguity_type
    result["dlite_ambiguity_score"] = dl["analyst"].ambiguity_score
    result["dlite_final_question"]  = dl_question
    _apply_scores(result, "dlite", query, ambiguity_type,
                  dl_question, gold_question, gold_answer, is_ambiguous, all_gold_questions)

    # ── System 3: ORACLE ─────────────────────────────────────────────────────
    result["oracle_is_ambiguous"]        = ora.is_ambiguous
    result["oracle_ambiguity_type"]      = ora.analyst.ambiguity_type
    result["oracle_ambiguity_score"]     = ora.analyst.ambiguity_score
    result["oracle_interp_a_confidence"] = ora.interpretation_a.confidence
    result["oracle_interp_b_confidence"] = ora.interpretation_b.confidence

    if ora.is_ambiguous and ora.clarifier:
        result["oracle_question_v1"]    = ora.clarifier.question_v1
        result["oracle_question_v2"]    = ora.clarifier.question_v2 or ""
        result["oracle_winner"]         = ora.judge.winner if ora.judge else ""
        result["oracle_final_question"] = ora.final_question or ""
    else:
        result["oracle_question_v1"]    = ""
        result["oracle_question_v2"]    = ""
        result["oracle_winner"]         = ""
        result["oracle_final_question"] = ""

    _apply_scores(result, "oracle", query, ambiguity_type,
                  ora.final_question or "", gold_question, gold_answer, is_ambiguous, all_gold_questions)

    return result


def run_evaluation(delay: float = 1.0):
    dataset   = load_csv()
    done      = _already_done()
    remaining = [r for r in dataset if r["id"] not in done]

    print(f"Total queries : {len(dataset)}")
    print(f"Already done  : {len(done)}")
    print(f"Remaining     : {len(remaining)}")
    print()

    f, writer = _open_writer()
    try:
        for i, row in enumerate(remaining, start=1):
            safe_query = row['query'][:70].encode("ascii", errors="replace").decode("ascii")
            print(f"[{i}/{len(remaining)}] {safe_query}...")
            try:
                result = _run_one(row)
            except Exception as e:
                print(f"  ERROR: {e}")
                result = {
                    "id":                     row["id"],
                    "query":                  row["query"],
                    "ground_truth_ambiguous": row["is_ambiguous"],
                    "gold_question":          row.get("gold_question", ""),
                    "gold_answer":            row.get("gold_answer", ""),
                    "error":                  traceback.format_exc(limit=2),
                    **{k: "" for k in FIELDNAMES
                       if k not in ("id", "query", "ground_truth_ambiguous",
                                    "gold_question", "gold_answer", "error")},
                }
            writer.writerow(result)
            f.flush()
            if i < len(remaining):
                time.sleep(delay)
    finally:
        f.close()
    print(f"\nDone. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
