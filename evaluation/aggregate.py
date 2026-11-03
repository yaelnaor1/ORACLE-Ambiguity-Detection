"""
Reads evaluation/results.csv and prints the final results table.
Covers all 3 scoring methods: Gemini Judge, Automatic (ROUGE/BERTScore/ARR), and flags.
"""
import csv
from pathlib import Path

RESULTS_PATH = Path("evaluation/results.csv")

SYSTEMS = {
    "Baseline GPT":    "baseline_gpt",
    "Baseline Claude": "baseline_claude",
    "Debate-Lite":     "dlite",
    "ORACLE":          "oracle",
}


def _f(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _b(val) -> bool | None:
    if str(val) in ("True", "true", "1"):  return True
    if str(val) in ("False", "false", "0"): return False
    return None

def avg(lst):
    vals = [v for v in lst if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else 0.0

def pct(lst):
    vals = [v for v in lst if v is not None]
    return round(sum(1 for v in vals if v) / len(vals) * 100, 1) if vals else 0.0


def aggregate():
    if not RESULTS_PATH.exists():
        print(f"No results file at {RESULTS_PATH}. Run evaluation/runner.py first.")
        return

    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]

    total = len(rows)
    if total == 0:
        print("No completed rows found.")
        return

    print(f"\nResults based on {total} completed queries")
    print(f"  Ambiguous: {sum(1 for r in rows if _b(r['ground_truth_ambiguous']))}")
    print(f"  Clear    : {sum(1 for r in rows if not _b(r['ground_truth_ambiguous']))}")

    # ── Method 1: Gemini Judge ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  METHOD 1 — Gemini 2.0 Flash Judge (Qualitative, 1-5 per axis)")
    print(f"{'='*90}")
    header = f"{'System':<20} {'Relevance':>10} {'Specificity':>12} {'Necessity':>10} {'Hallucin%':>10} {'FalsePos%':>10}"
    print(header)
    print("-" * len(header))
    for label, px in SYSTEMS.items():
        rel  = avg([_f(r[f"{px}_relevance"])    for r in rows])
        spec = avg([_f(r[f"{px}_specificity"])  for r in rows])
        nec  = avg([_f(r[f"{px}_necessity"])    for r in rows])
        hall = pct([_b(r[f"{px}_hallucination"])  for r in rows])
        fp   = pct([_b(r[f"{px}_false_positive"]) for r in rows])
        print(f"{label:<20} {rel:>10} {spec:>12} {nec:>10} {hall:>9}% {fp:>9}%")

    # ── Method 2: Automatic ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  METHOD 2 — Automatic Reference-Based Scoring")
    print(f"{'='*75}")
    header2 = f"{'System':<20} {'ROUGE-L':>10} {'BERTScore':>12} {'ARR%':>8}"
    print(header2)
    print("-" * len(header2))
    for label, px in SYSTEMS.items():
        rl  = avg([_f(r[f"{px}_rouge_l"])   for r in rows])
        bs  = avg([_f(r[f"{px}_bertscore"]) for r in rows])
        arr = pct([_b(r[f"{px}_arr"])       for r in rows if r[f"{px}_arr"] != ""])
        print(f"{label:<20} {rl:>10} {bs:>12} {arr:>7}%")

    # ── Combined summary (matches the plan's results table) ───────────────────
    print(f"\n{'='*105}")
    print("  COMBINED RESULTS TABLE")
    print(f"{'='*105}")
    header3 = (f"{'System':<20} {'Rel':>5} {'Spec':>5} {'Nec':>5} "
               f"{'ROUGE-L':>8} {'BERT':>6} {'ARR%':>6} {'Hall%':>7} {'FP%':>5}")
    print(header3)
    print("-" * len(header3))
    for label, px in SYSTEMS.items():
        rel  = avg([_f(r[f"{px}_relevance"])      for r in rows])
        spec = avg([_f(r[f"{px}_specificity"])    for r in rows])
        nec  = avg([_f(r[f"{px}_necessity"])      for r in rows])
        rl   = avg([_f(r[f"{px}_rouge_l"])        for r in rows])
        bs   = avg([_f(r[f"{px}_bertscore"])      for r in rows])
        arr  = pct([_b(r[f"{px}_arr"])            for r in rows if r[f"{px}_arr"] != ""])
        hall = pct([_b(r[f"{px}_hallucination"])  for r in rows])
        fp   = pct([_b(r[f"{px}_false_positive"]) for r in rows])
        print(f"{label:<20} {rel:>5} {spec:>5} {nec:>5} "
              f"{rl:>8} {bs:>6} {arr:>5}% {hall:>6}% {fp:>4}%")

    # ── ORACLE breakdown by ambiguity type ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ORACLE — breakdown by ambiguity type (Gemini judge scores)")
    print(f"{'='*70}")
    h = f"{'Type':<12} {'N':>5} {'Relevance':>10} {'Specificity':>12} {'Necessity':>10}"
    print(h)
    print("-" * len(h))
    for t in ["Semantic", "Generalize", "Specify", "None"]:
        sub = [r for r in rows if r.get("oracle_ambiguity_type") == t]
        if not sub:
            continue
        rel  = avg([_f(r["oracle_relevance"])   for r in sub])
        spec = avg([_f(r["oracle_specificity"]) for r in sub])
        nec  = avg([_f(r["oracle_necessity"])   for r in sub])
        print(f"{t:<12} {len(sub):>5} {rel:>10} {spec:>12} {nec:>10}")


if __name__ == "__main__":
    aggregate()
