"""
Ambiguity-Detection Metrics (Method 4).

The core task is a binary classification: given a query, should the system ask a
clarifying question (ambiguous) or answer directly (clear)? Methods 1-3 score the
*quality* of the clarifying question, but only on the subset of queries a system
chose to ask about. This module measures the prior, unconditional question: how
well does each system *detect* ambiguity in the first place?

Everything here is computed from the already-saved evaluation/results.csv — no
pipeline re-run and no API calls. The continuous analyst `*_ambiguity_score`
columns let us re-threshold the decision offline (the 0.5 cutoff was applied at
run time, but the raw score was saved), which is what powers the threshold sweep
and the threshold-independent ROC/PR-AUC summary.

Only ORACLE and Debate-Lite appear here: they are the systems that explicitly
classify ambiguity. The baselines answer directly by design and never abstain,
so a detection confusion matrix does not apply to them.

Usage:
  python -m evaluation.detection            # full report (table + sweep + AUC)
  python -m evaluation.detection sweep      # just the threshold sweep
"""
import csv
import sys
from pathlib import Path

from oracle.config import AMBIGUITY_THRESHOLD

RESULTS_PATH = Path("evaluation/results.csv")

# label -> (predicted-bool column, continuous-score column)
SYSTEMS = {
    "Debate-Lite": ("dlite_is_ambiguous",  "dlite_ambiguity_score"),
    "ORACLE":      ("oracle_is_ambiguous", "oracle_ambiguity_score"),
}

SWEEP_THRESHOLDS = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7]


# ── parsing helpers ───────────────────────────────────────────────────────────

def _b(v) -> bool:
    return str(v).strip().lower() in ("true", "1")

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _load_rows() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found — run evaluation/runner.py first.")
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not r.get("error")]


# ── core metric ───────────────────────────────────────────────────────────────

def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = list of (ground_truth_ambiguous, predicted_ambiguous)."""
    tp = sum(1 for gt, p in pairs if gt and p)
    fp = sum(1 for gt, p in pairs if not gt and p)
    fn = sum(1 for gt, p in pairs if gt and not p)
    tn = sum(1 for gt, p in pairs if not gt and not p)
    n  = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / n if n else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(f1, 3), "accuracy": round(accuracy, 3),
    }


def detection_at_threshold(rows: list[dict], score_col: str, threshold: float) -> dict:
    """Re-derive the ambiguous/clear decision from the saved continuous score."""
    pairs = [
        (_b(r["ground_truth_ambiguous"]), _f(r[score_col]) > threshold)
        for r in rows if _f(r[score_col]) is not None
    ]
    return confusion(pairs)


def detection_as_run(rows: list[dict], pred_col: str) -> dict:
    """Detection using the system's own saved decision (the 0.5 cutoff as run)."""
    pairs = [(_b(r["ground_truth_ambiguous"]), _b(r[pred_col])) for r in rows]
    return confusion(pairs)


def auc_scores(rows: list[dict], score_col: str) -> dict | None:
    """Threshold-independent ROC-AUC and PR-AUC over the continuous score."""
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        return None
    y_true, y_score = [], []
    for r in rows:
        s = _f(r[score_col])
        if s is None:
            continue
        y_true.append(1 if _b(r["ground_truth_ambiguous"]) else 0)
        y_score.append(s)
    if len(set(y_true)) < 2:
        return None
    return {
        "roc_auc": round(roc_auc_score(y_true, y_score), 3),
        "pr_auc":  round(average_precision_score(y_true, y_score), 3),
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]):
    print(f"\n{'='*78}")
    print(f"  AMBIGUITY DETECTION -- confusion matrix and P/R/F1 (threshold={AMBIGUITY_THRESHOLD})")
    print(f"{'='*78}")
    n_amb = sum(1 for r in rows if _b(r["ground_truth_ambiguous"]))
    print(f"  Evaluated on {len(rows)} queries  ({n_amb} ambiguous, {len(rows)-n_amb} clear)\n")
    hdr = f"{'System':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, (pred_col, _) in SYSTEMS.items():
        m = detection_as_run(rows, pred_col)
        print(f"{label:<14} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4} "
              f"{m['precision']:>7} {m['recall']:>7} {m['f1']:>7} {m['accuracy']:>7}")
    print("\n  Note: high precision, low recall -- both systems miss most ambiguous")
    print("  queries (large FN column). See the threshold sweep below.")


def _print_sweep(rows: list[dict]):
    print(f"\n{'='*78}")
    print("  THRESHOLD SWEEP -- re-thresholding the saved analyst ambiguity score")
    print(f"{'='*78}")
    for label, (_, score_col) in SYSTEMS.items():
        print(f"\n  {label}")
        print(f"  {'thresh':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7}")
        print(f"  {'-'*39}")
        best = (None, -1.0)
        for t in SWEEP_THRESHOLDS:
            m = detection_at_threshold(rows, score_col, t)
            if m["f1"] > best[1]:
                best = (t, m["f1"])
            mark = "  <- as run" if abs(t - AMBIGUITY_THRESHOLD) < 1e-9 else ""
            print(f"  {t:>7} {m['precision']:>7} {m['recall']:>7} {m['f1']:>7} {m['accuracy']:>7}{mark}")
        print(f"  best F1 = {best[1]:.3f} at threshold = {best[0]}  "
              f"(vs F1 at {AMBIGUITY_THRESHOLD}: only marginally lower)")


def _print_auc(rows: list[dict]):
    print(f"\n{'='*78}")
    print("  THRESHOLD-INDEPENDENT SUMMARY -- area under ROC / PR curve")
    print(f"{'='*78}")
    print("  (0.5 ROC-AUC = no better than chance; 1.0 = perfect ranking)\n")
    print(f"  {'System':<14} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print(f"  {'-'*34}")
    for label, (_, score_col) in SYSTEMS.items():
        a = auc_scores(rows, score_col)
        if a is None:
            print(f"  {label:<14} {'n/a':>9} {'n/a':>9}")
        else:
            print(f"  {label:<14} {a['roc_auc']:>9} {a['pr_auc']:>9}")


def report():
    rows = _load_rows()
    _print_table(rows)
    _print_sweep(rows)
    _print_auc(rows)
    print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    rows = _load_rows()
    if cmd == "sweep":
        _print_sweep(rows)
    elif cmd == "auc":
        _print_auc(rows)
    else:
        report()
