"""
Loads sewon/ambig_qa from HuggingFace and saves a clean CSV for the 3 systems.

Columns in output CSV:
  id            - original dataset id
  query         - the original ambiguous/clear question
  is_ambiguous  - True/False (ground truth label)
  gold_question - gold clarifying sub-question (first one, if ambiguous; else empty)
  all_gold_questions - all gold sub-questions as a pipe-separated string
"""
import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "ambig_qa.csv"

# How many samples to pull per class (keep costs manageable during dev)
MAX_AMBIGUOUS = 333
MAX_CLEAR = 167


def load_and_save(max_ambiguous: int = MAX_AMBIGUOUS, max_clear: int = MAX_CLEAR) -> Path:
    from datasets import load_dataset

    print("Downloading sewon/ambig_qa (train split) ...")
    ds = load_dataset("sewon/ambig_qa", "light", split="train", token=HF_TOKEN)

    rows = []
    ambiguous_count = 0
    clear_count = 0

    for item in ds:
        annotations = item.get("annotations", {})
        ann_type = annotations.get("type", [])

        # Each item can have multiple annotations; take the majority vote
        is_ambiguous = ann_type.count("multipleQAs") > ann_type.count("singleAnswer")

        if is_ambiguous and ambiguous_count >= max_ambiguous:
            continue
        if not is_ambiguous and clear_count >= max_clear:
            continue

        # Extract gold clarifying questions + answers from qaPairs
        # Structure: qaPairs = [{"question": ["q1", "q2", ...], "answer": [["a1"], ["a2"]]}]
        gold_questions = []
        gold_answers   = []
        qa_pairs_list  = annotations.get("qaPairs", [])
        for qa_pair in qa_pairs_list:
            questions = qa_pair.get("question", [])
            answers   = qa_pair.get("answer", [])
            for i, q in enumerate(questions):
                q = q.strip()
                if q and q not in gold_questions:
                    gold_questions.append(q)
                    # answers[i] is a list of strings; take the first
                    ans = answers[i][0] if i < len(answers) and answers[i] else ""
                    gold_answers.append(ans.strip())

        rows.append({
            "id":                item["id"],
            "query":             item["question"],
            "is_ambiguous":      is_ambiguous,
            "gold_question":     gold_questions[0] if gold_questions else "",
            "gold_answer":       gold_answers[0]   if gold_answers   else "",
            "all_gold_questions": " | ".join(gold_questions),
        })

        if is_ambiguous:
            ambiguous_count += 1
        else:
            clear_count += 1

        if ambiguous_count >= max_ambiguous and clear_count >= max_clear:
            break

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows -> {OUTPUT_PATH}")
    print(f"  Ambiguous : {ambiguous_count}")
    print(f"  Clear     : {clear_count}")
    return OUTPUT_PATH


def load_csv() -> list[dict]:
    """Load the saved CSV back into a list of dicts."""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} not found. Run dataset_loader.py first to download the data."
        )
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        row["is_ambiguous"] = row["is_ambiguous"] == "True"
    return rows


if __name__ == "__main__":
    load_and_save()
