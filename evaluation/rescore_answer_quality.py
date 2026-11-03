"""
Add new answer-quality metrics to existing results.csv without re-running the pipeline.

New columns added per system:
  - {prefix}_exact_match   : bool — strict normalized question match vs ANY gold question
  - {prefix}_nearby_match  : bool — BERTScore >= 0.85 vs ANY gold question (free, no API)
  - {prefix}_answer_match  : "exact"|"close"|"far" — final answer quality vs gold answer (GPT call)

Question matches are cross-validated against the *full* set of gold sub-questions
(`all_gold_questions` in data/ambig_qa.csv) rather than just the first one — AmbigQA
frequently lists several equally-valid disambiguations, so comparing against only one
of them would unfairly penalize a system that produced a different valid gold question.

Saves to evaluation/results_enriched.csv
"""
import csv
import time
from pathlib import Path

from evaluation.auto_scorer import exact_match_multi, nearby_match_multi, answer_match_score, split_gold_questions

INPUT_PATH     = Path("evaluation/results.csv")
DATASET_PATH   = Path("data/ambig_qa.csv")
OUTPUT_PATH    = Path("evaluation/results_enriched.csv")


def _load_all_gold_questions() -> dict[str, str]:
    """id -> pipe-separated all_gold_questions, from the original dataset CSV."""
    if not DATASET_PATH.exists():
        return {}
    return {r["id"]: r.get("all_gold_questions", "")
            for r in csv.DictReader(open(DATASET_PATH, encoding="utf-8"))}

SYSTEMS = [
    ("baseline_gpt",    "baseline_gpt_response"),
    ("baseline_claude", "baseline_claude_response"),
    ("dlite",           "dlite_final_question"),
    ("oracle",          "oracle_final_question"),
]

NEW_COLS = [
    f"{prefix}_{metric}"
    for prefix, _ in SYSTEMS
    for metric in ("exact_match", "nearby_match", "answer_match")
]


def rescore(delay: float = 0.3):
    rows = list(csv.DictReader(open(INPUT_PATH, encoding="utf-8")))
    total = len(rows)
    gold_questions_by_id = _load_all_gold_questions()

    already_done: set[str] = set()
    if OUTPUT_PATH.exists():
        already_done = {r["id"] for r in csv.DictReader(open(OUTPUT_PATH, encoding="utf-8"))
                        if r.get("oracle_answer_match")}

    out_fieldnames = list(rows[0].keys()) + NEW_COLS
    is_new = not OUTPUT_PATH.exists()
    out_f  = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=out_fieldnames, extrasaction="ignore")
    if is_new:
        writer.writeheader()

    try:
        for i, row in enumerate(rows, 1):
            if row["id"] in already_done:
                print(f"[{i}/{total}] skip {row['id']}")
                continue

            is_ambiguous = str(row.get("ground_truth_ambiguous", "")).lower() in ("true", "1")
            gold_question = row.get("gold_question", "")
            gold_answer   = row.get("gold_answer", "")
            query         = row["query"]

            references = (split_gold_questions(gold_questions_by_id.get(row["id"], ""))
                          or [gold_question])

            safe_query = query.encode("ascii", errors="replace").decode("ascii")
            print(f"[{i}/{total}] {safe_query[:70]}...")

            for prefix, question_col in SYSTEMS:
                question = row.get(question_col, "")

                if not is_ambiguous or not gold_question:
                    row[f"{prefix}_exact_match"]  = ""
                    row[f"{prefix}_nearby_match"] = ""
                    row[f"{prefix}_answer_match"] = ""
                    continue

                row[f"{prefix}_exact_match"]  = exact_match_multi(question, references) if question else ""
                row[f"{prefix}_nearby_match"] = nearby_match_multi(question, references) if question else ""

                if not question:
                    row[f"{prefix}_answer_match"] = ""
                    continue

                for attempt in range(3):
                    try:
                        row[f"{prefix}_answer_match"] = answer_match_score(query, question, gold_answer)
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"  ERROR answer_match {prefix}: {e}")
                            row[f"{prefix}_answer_match"] = ""
                        else:
                            import time as _t; _t.sleep(2)

            writer.writerow(row)
            out_f.flush()
            if i < total:
                time.sleep(delay)
    finally:
        out_f.close()

    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    rescore()
