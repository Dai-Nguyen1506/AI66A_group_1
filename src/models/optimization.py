import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, f1_score, fbeta_score,
    average_precision_score, confusion_matrix,
    precision_score, recall_score
)

def find_optimal_threshold(y_true, y_prob, beta=2, plot=True):
    """
    Evaluates all thresholds on the PR curve.

    beta: weight of recall relative to precision in F-beta score.
          beta=1  -> F1 (balanced)
          beta=2  -> penalises missed fraud more (recommended for fraud)
          beta=0.5-> penalises false alerts more
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # ── compute F-beta for every threshold ────────────────────────────────────
    # precision/recall arrays have one extra element (boundary), align them
    p = precision[:-1]
    r = recall[:-1]

    # avoid division by zero
    denom = (beta**2 * p) + r
    fbeta  = np.where(denom > 0,
                      (1 + beta**2) * p * r / denom,
                      0)

    best_idx       = np.argmax(fbeta)
    best_threshold = thresholds[best_idx]
    best_fbeta     = fbeta[best_idx]

    # ── summary table ─────────────────────────────────────────────────────────
    results = pd.DataFrame({
        'threshold' : np.round(thresholds, 4),
        'precision' : np.round(p, 4),
        'recall'    : np.round(r, 4),
        f'f{beta}'  : np.round(fbeta, 4),
    })

    print(f"\n{'='*55}")
    print(f"  Optimal threshold (F{beta} score)")
    print(f"{'='*55}")
    print(f"  Threshold : {best_threshold:.4f}")
    print(f"  Precision : {p[best_idx]:.4f}")
    print(f"  Recall    : {r[best_idx]:.4f}")
    print(f"  F{beta} score : {best_fbeta:.4f}")
    print(f"{'='*55}\n")

    # ── confusion matrix at optimal threshold ─────────────────────────────────
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix @ threshold={best_threshold:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Fraud caught      : {tp/(tp+fn)*100:.1f}%  (recall)")
    print(f"  Alert precision   : {tp/(tp+fp)*100:.1f}%  (precision)")
    print(f"  False alert rate  : {fp/(fp+tn)*100:.2f}% of legit txns flagged\n")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PR curve
        axes[0].plot(recall, precision, color='steelblue', lw=2)
        axes[0].scatter(r[best_idx], p[best_idx],
                        color='red', zorder=5, s=100,
                        label=f'Best threshold={best_threshold:.3f}')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.4)

        # F-beta vs threshold
        axes[1].plot(thresholds, fbeta, color='darkorange', lw=2)
        axes[1].axvline(best_threshold, color='red', linestyle='--',
                        label=f'Best={best_threshold:.3f}  F{beta}={best_fbeta:.4f}')
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel(f'F{beta} Score')
        axes[1].set_title(f'F{beta} Score vs Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.4)

        plt.tight_layout()
        plt.show()

    return best_threshold, results

def eval_at_threshold(y_true, y_prob, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n── {label} (threshold={threshold:.4f}) ──")
    print(f"  PR-AUC    : {average_precision_score(y_true, y_prob):.5f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F2        : {fbeta_score(y_true, y_pred, beta=2):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")