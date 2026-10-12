"""Metrics, computed downstream from stored predictions.

Because every metric reads the same prediction file, the metric factor costs no
extra compute and is guaranteed to see identical predictions across metrics --
which is what makes "metric choice" a clean factor in the decomposition.
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm
from scipy import stats

HIGHER_IS_BETTER = {
    "auroc": True, "auprc": True, "mcc": True, "bacc": True, "f1": True,
    "accuracy": True, "spearman": True, "pearson": True, "r2": True,
    "rmse": False, "mae": False,
}


def classification_scores(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    out: dict[str, float] = {}
    # Guard the degenerate single-class fold rather than letting sklearn raise
    # mid-sweep; a NaN here is informative and survives into the results table.
    if len(np.unique(y_true)) < 2:
        return {k: float("nan") for k in ["auroc", "auprc", "mcc", "bacc", "f1", "accuracy"]}

    out["auroc"] = float(skm.roc_auc_score(y_true, y_score))
    out["auprc"] = float(skm.average_precision_score(y_true, y_score))
    out["mcc"] = float(skm.matthews_corrcoef(y_true, y_pred))
    out["bacc"] = float(skm.balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(skm.f1_score(y_true, y_pred, zero_division=0))
    out["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
    return out


def regression_scores(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "spearman": float(stats.spearmanr(y_true, y_pred).statistic),
        "pearson": float(stats.pearsonr(y_true, y_pred).statistic),
        "r2": float(skm.r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "mae": float(skm.mean_absolute_error(y_true, y_pred)),
    }
