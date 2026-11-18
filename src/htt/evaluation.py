import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .modeling import CLASSES, REWARD


def avg_reward(y, pred, R=REWARD):
    return float(R[np.asarray(pred), np.asarray(y)].mean())


def bnr(y, pred, R=REWARD):
    # Balanced Normalized Reward (Hard to Park, Sec. 7.2): mean reward after giving every TRUE class equal weight.
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([R[pred[y == c], c].mean() for c in np.unique(y)]))


def evaluate(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                avg_reward=avg_reward(y, pred), bnr=bnr(y, pred),
                frac_easy=(pred == 0).mean(), frac_medium=(pred == 1).mean(), frac_hard=(pred == 2).mean(),
                costly_errors=float(((pred == 0) & (y == 2)).mean()))  # predicted easy, actually hard


def compare(preds: dict, y, ref: str = None) -> pd.DataFrame:
    # preds: {name: predictions}. Adds relative BNR vs `ref` (same idea as the paper's relative BNR).
    t = pd.DataFrame({k: evaluate(y, v) for k, v in preds.items()}).T
    if ref is not None:
        t["rel_bnr"] = t["bnr"] - t.loc[ref, "bnr"]
    return t


def week_bootstrap(y, pred_a, pred_b, weeks, n=1000, seed=0):
    # Block bootstrap over calendar weeks: CI for BNR(a) - BNR(b).
    y, a, b, weeks = map(np.asarray, (y, pred_a, pred_b, weeks))
    rng = np.random.default_rng(seed)
    uw = np.unique(weeks)
    idx_by_w = {w: np.where(weeks == w)[0] for w in uw}
    diffs = []
    for _ in range(n):
        ii = np.concatenate([idx_by_w[w] for w in rng.choice(uw, len(uw))])
        diffs.append(bnr(y[ii], a[ii]) - bnr(y[ii], b[ii]))
    return float(np.mean(diffs)), tuple(np.percentile(diffs, [2.5, 97.5]))


def plot_confusion(y, pred, title="", ax=None):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=[0, 1, 2], normalize="true")
    ax = ax or plt.gca()
    ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
