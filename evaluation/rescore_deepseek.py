"""
Re-score existing results.csv with DeepSeek as an additional judge.
Reads the already-saved questions — no pipeline re-run needed.
Saves to evaluation/results_with_deepseek.csv
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from oracle.agents import deepseek_eval_judge

INPUT_PATH  = Path("evaluation/results.csv")
OUTPUT_PATH = Path("evaluation/results_with_deepseek.csv")

SYSTEMS = [
    ("baseline_gpt",    "baseline_gpt_response"),
    ("baseline_claude", "baseline_claude_response"),
    ("dlite",           "dlite_final_question"),
    ("oracle",          "oracle_final_question"),
]

DEEPSEEK_SCORE_FIELDS = [
    "ds_relevance", "ds_specificity", "ds_necessity",
    "ds_hallucination", "ds_false_positive", "ds_explanation",
]


def _deepseek_fields(prefix: str) -> list[str]:
    return [f"{prefix}_{f}" for f in DEEPSEEK_SCORE_FIELDS]


def _score_one(prefix: str, question_col: str, row: dict, ambiguity_type: str) -> dict:
    question = row.get(question_col, "")
    for attempt in range(1, 6):
        try:
            s = deepseek_eval_judge(row["query"], ambiguity_type, question)
            return {
                f"{prefix}_ds_relevance":      s.relevance,
                f"{prefix}_ds_specificity":    s.specificity,
                f"{prefix}_ds_necessity":      s.necessity,
                f"{prefix}_ds_hallucination":  s.hallucination,
                f"{prefix}_ds_false_positive": s.false_positive,
                f"{prefix}_ds_explanation":    s.explanation,
            }
        except Exception as e:
            wait = attempt * 10
            print(f"  RETRY {prefix} (attempt {attempt}/5, waiting {wait}s): {e}")
            time.sleep(wait)
    print(f"  FAILED {prefix} after 5 attempts — skipping")
    return {f: "" for f in _deepseek_fields(prefix)}


def rescore():
    rows = list(csv.DictReader(open(INPUT_PATH, encoding="utf-8")))
    total = len(rows)

    new_cols = [f for prefix, _ in SYSTEMS for f in _deepseek_fields(prefix)]
    out_fieldnames = list(rows[0].keys()) + new_cols

    already_done: set[str] = set()
    if OUTPUT_PATH.exists():
        already_done = {r["id"] for r in csv.DictReader(open(OUTPUT_PATH, encoding="utf-8"))
                        if r.get("baseline_gpt_ds_relevance")}

    is_new = not OUTPUT_PATH.exists()
    out_f  = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=out_fieldnames, extrasaction="ignore")
    if is_new:
        writer.writeheader()

    try:
        for i, row in enumerate(rows, 1):
            if row["id"] in already_done:
                continue

            safe_q = row["query"].encode("ascii", errors="replace").decode("ascii")
            print(f"[{i}/{total}] {safe_q[:70]}...")
            ambiguity_type = row.get("oracle_ambiguity_type") or "None"

            # Score all 4 systems in parallel
            scores = {}
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = {ex.submit(_score_one, prefix, col, row, ambiguity_type): prefix
                           for prefix, col in SYSTEMS}
                for fut in as_completed(futures):
                    result = fut.result()
                    row.update(result)
                    scores[futures[fut]] = result

            for prefix, _ in SYSTEMS:
                s = scores.get(prefix, {})
                rel  = s.get(f"{prefix}_ds_relevance",      "?")
                spec = s.get(f"{prefix}_ds_specificity",    "?")
                nec  = s.get(f"{prefix}_ds_necessity",      "?")
                hall = s.get(f"{prefix}_ds_hallucination",  "?")
                fp   = s.get(f"{prefix}_ds_false_positive", "?")
                print(f"  {prefix:<16} rel={rel} spec={spec} nec={nec} hall={hall} fp={fp}")

            writer.writerow(row)
            out_f.flush()
    finally:
        out_f.close()

    print(f"\nDone. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    rescore()
