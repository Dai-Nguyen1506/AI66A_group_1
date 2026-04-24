import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

def find_optimal_threshold(model, X_test, y_test, beta=2, name="Model", plot=True, plot_saving=False):
    """
    Evaluates all thresholds on the PR curve.

    beta: weight of recall relative to precision in F-beta score.
          beta=1  -> F1 (balanced)
          beta=2  -> penalises missed fraud more (recommended for fraud)
          beta=0.5-> penalises false alerts more
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # compute F-beta for every threshold
    # precision/recall arrays have one extra element (boundary), align them
    p = precision[:-1]
    r = recall[:-1]

    # avoid division by zero
    denom = (beta**2 * p) + r
    fbeta  = np.where(denom > 0, (1 + beta**2) * p * r / denom, 0)

    best_idx       = np.argmax(fbeta)
    best_threshold = thresholds[best_idx]
    best_fbeta     = fbeta[best_idx]

    print(f"{'='*50}")
    print(f"  Optimal threshold (F{beta} score) for {name}")
    print(f"{'='*50}")
    print(f"  Threshold : {best_threshold:.4f}")
    print(f"  Precision : {p[best_idx]:.4f}")
    print(f"  Recall    : {r[best_idx]:.4f}")
    print(f"  F{beta} score : {best_fbeta:.4f}")
    print(f"{'='*50}")

    # confusion matrix at optimal threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    print(f"  Confusion Matrix @ threshold={best_threshold:.4f}")
    print(confusion_matrix(y_test, y_pred_opt))

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

        fig.suptitle(f"{name} Threshold Optimization", fontsize=16, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if plot_saving:
            plt.savefig(f"../reports/{name}_optimization.png", dpi=300, bbox_inches='tight')
        plt.show()

    return best_threshold