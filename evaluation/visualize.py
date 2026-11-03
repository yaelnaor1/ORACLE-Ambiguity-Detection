"""
Ambiguity-detection plots.

Reads the saved evaluation/results.csv and reuses the pure metric functions from
evaluation/detection.py — no embeddings, no API calls. Three figures:

Usage:
  python -m evaluation.visualize confusion  # detection confusion matrices
  python -m evaluation.visualize f1curve    # detection F1 vs threshold
  python -m evaluation.visualize detection  # detection P/R/F1 bar chart

Requires: pip install matplotlib
"""
import sys
from pathlib import Path

PLOTS_DIR = Path("evaluation/plots")


def cmd_confusion():
    """Side-by-side 2x2 confusion-matrix heatmaps for ORACLE and Debate-Lite."""
    import numpy as np
    import matplotlib.pyplot as plt
    from evaluation.detection import _load_rows, detection_as_run, SYSTEMS as DET_SYSTEMS

    rows = _load_rows()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(DET_SYSTEMS), figsize=(5 * len(DET_SYSTEMS), 4.5))
    if len(DET_SYSTEMS) == 1:
        axes = [axes]

    for ax, (label, (pred_col, _)) in zip(axes, DET_SYSTEMS.items()):
        m = detection_as_run(rows, pred_col)
        # rows = actual (Ambiguous, Clear); cols = predicted (Ambiguous, Clear)
        mat = np.array([[m["tp"], m["fn"]],
                        [m["fp"], m["tn"]]])
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred:\nAmbiguous", "Pred:\nClear"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual:\nAmbiguous", "Actual:\nClear"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=16,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_title(f"{label}\nF1={m['f1']:.3f}  Recall={m['recall']:.3f}  Prec={m['precision']:.3f}")

    fig.suptitle("Ambiguity-Detection Confusion Matrices\n"
                 "(large top-right cell = ambiguous queries missed = low recall)",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def cmd_f1curve():
    """F1 vs decision threshold, with the 0.5 default marked."""
    import numpy as np
    import matplotlib.pyplot as plt
    from oracle.config import AMBIGUITY_THRESHOLD
    from evaluation.detection import _load_rows, detection_at_threshold, SYSTEMS as DET_SYSTEMS

    rows = _load_rows()
    thresholds = np.linspace(0.0, 1.0, 101)
    # Below this threshold the classifier degenerates toward "always ask"
    # (recall -> 1, precision -> base rate); F1 there is the trivial baseline,
    # not a usable operating point, so we exclude it from the "best" marker.
    OPERATING_MIN = 0.1
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = {"ORACLE": "#C44E52", "Debate-Lite": "#55A868"}
    for label, (_, score_col) in DET_SYSTEMS.items():
        f1s = [detection_at_threshold(rows, score_col, t)["f1"] for t in thresholds]
        # best F1 within the usable operating range only
        usable = [(t, f) for t, f in zip(thresholds, f1s) if t >= OPERATING_MIN]
        best_t, best_f = max(usable, key=lambda tf: tf[1])
        color = palette.get(label, None)
        f1_default = detection_at_threshold(rows, score_col, AMBIGUITY_THRESHOLD)["f1"]
        ax.plot(thresholds, f1s, color=color, linewidth=2,
                label=f"{label}: F1={f1_default:.3f} @ {AMBIGUITY_THRESHOLD} "
                      f"(best usable {best_f:.3f} @ {best_t:.2f})")
        ax.scatter([best_t], [best_f], color=color, zorder=5, s=40)

    ax.axvline(AMBIGUITY_THRESHOLD, color="gray", linestyle="--",
               label=f"Default threshold = {AMBIGUITY_THRESHOLD}")
    ax.axvspan(0.0, OPERATING_MIN, color="gray", alpha=0.12)
    ax.text(0.005, 0.02, "trivial\n'always ask'\nregion", fontsize=8,
            color="gray", va="bottom", ha="left", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Ambiguity-score decision threshold")
    ax.set_ylabel("F1 (detection)")
    ax.set_title("Detection F1 vs Threshold\n"
                 "Flat in the usable range = re-thresholding barely helps "
                 "(analyst under-scores ambiguity)")
    ax.set_ylim(0, max(0.8, max(f1s) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    out = PLOTS_DIR / "f1_vs_threshold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def cmd_detection_bars():
    """Grouped bar chart of Precision / Recall / F1 per system (the headline figure)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from evaluation.detection import _load_rows, detection_as_run, SYSTEMS as DET_SYSTEMS

    rows = _load_rows()
    metrics = ["precision", "recall", "f1"]
    labels  = list(DET_SYSTEMS.keys())
    data = {lab: detection_as_run(rows, col) for lab, (col, _) in DET_SYSTEMS.items()}

    x = np.arange(len(metrics))
    width = 0.8 / len(labels)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {"ORACLE": "#C44E52", "Debate-Lite": "#55A868"}

    for i, lab in enumerate(labels):
        vals = [data[lab][mname] for mname in metrics]
        bars = ax.bar(x + i * width, vals, width, label=lab, color=palette.get(lab))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ambiguity-Detection Metrics by System\n(high precision, low recall)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out = PLOTS_DIR / "detection_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "confusion":
        cmd_confusion()
    elif cmd == "f1curve":
        cmd_f1curve()
    elif cmd == "detection":
        cmd_detection_bars()
    else:
        print("Usage:")
        print("  python -m evaluation.visualize confusion  # detection confusion matrices")
        print("  python -m evaluation.visualize f1curve    # detection F1 vs threshold")
        print("  python -m evaluation.visualize detection  # detection P/R/F1 bar chart")
