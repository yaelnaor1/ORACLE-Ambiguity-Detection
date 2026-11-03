"""
ICC (Intraclass Correlation Coefficient) — inter-judge agreement between the two
automated LLM judges (Gemini and DeepSeek) scoring the same items.

Run after rescore_deepseek.py to compare Gemini vs DeepSeek on the same queries.

Usage:
  python -m evaluation.icc gemini_vs_deepseek   # ICC between Gemini and DeepSeek judges
"""
import csv
import sys
from pathlib import Path

ENRICHED_PATH = Path("evaluation/results_with_deepseek.csv")

SYSTEMS = ["baseline_gpt", "baseline_claude", "dlite", "oracle"]
AXES    = ["relevance", "specificity", "necessity"]


# ── ICC(2,1) — two-way random, single measures ───────────────────────────────

def icc_two_way(rater_a: list[float], rater_b: list[float]) -> float:
    """ICC(2,1): two-way random effects model, single measures."""
    n = len(rater_a)
    if n < 2:
        return float("nan")

    mean_a = sum(rater_a) / n
    mean_b = sum(rater_b) / n
    grand  = (mean_a + mean_b) / 2

    # Between-subjects (rows) mean square
    row_means = [(a + b) / 2 for a, b in zip(rater_a, rater_b)]
    ss_rows   = 2 * sum((m - grand) ** 2 for m in row_means)
    ms_rows   = ss_rows / (n - 1)

    # Between-raters (columns) mean square
    ss_cols = n * ((mean_a - grand) ** 2 + (mean_b - grand) ** 2)
    ms_cols = ss_cols  # df = 1

    # Error mean square
    ss_err = sum((a - row_means[i]) ** 2 + (b - row_means[i]) ** 2
                 for i, (a, b) in enumerate(zip(rater_a, rater_b)))
    ms_err = ss_err / ((n - 1) * 1)  # df = (n-1)*(k-1), k=2

    denom = ms_rows + ms_cols / n - ms_err / n
    if denom == 0:
        return float("nan")
    return round((ms_rows - ms_err) / denom, 4)


# ── Command ──────────────────────────────────────────────────────────────────

def cmd_gemini_vs_deepseek():
    if not ENRICHED_PATH.exists():
        print(f"Run evaluation/rescore_deepseek.py first to generate {ENRICHED_PATH}")
        return

    rows = [r for r in csv.DictReader(open(ENRICHED_PATH, encoding="utf-8")) if not r.get("error")]
    print(f"\nICC(2,1) -- Gemini Judge vs DeepSeek Judge  (n={len(rows)} queries)")
    print(f"{'System':<20} {'Axis':<14} {'ICC':>8}  Interpretation")
    print("-" * 65)

    def interp(v):
        if v >= 0.90: return "excellent"
        if v >= 0.75: return "good"
        if v >= 0.50: return "moderate"
        return "poor"

    def safe_float(v):
        try: return float(v)
        except: return None

    for px in SYSTEMS:
        for axis in AXES:
            gemini_col   = f"{px}_{axis}"
            deepseek_col = f"{px}_ds_{axis}"
            pairs = [
                (safe_float(r[gemini_col]), safe_float(r[deepseek_col]))
                for r in rows
                if safe_float(r.get(gemini_col)) is not None
                and safe_float(r.get(deepseek_col)) is not None
            ]
            if not pairs:
                continue
            a_vals, b_vals = zip(*pairs)
            val = icc_two_way(list(a_vals), list(b_vals))
            print(f"{px:<20} {axis:<14} {val:>8}  {interp(val)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "gemini_vs_deepseek":
        cmd_gemini_vs_deepseek()
    else:
        print("Usage:")
        print("  python -m evaluation.icc gemini_vs_deepseek   # ICC: Gemini vs DeepSeek")
