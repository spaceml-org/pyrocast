from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

from pyrocast.utils.data.cube_matching import NRL_STATE_NAMES


def auc(ytest, ypred):
    return roc_auc_score(ytest, ypred)


def auc_or_nan(y_true, y_pred_proba) -> float:
    """ROC AUC, or NaN when y_true holds a single class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred_proba))


def classification_report(y_true, y_pred_proba, threshold=0.5):
    """Compute AUC, FPR, and FNR from predicted probabilities.

    Args:
        y_true: true binary labels
        y_pred_proba: predicted probabilities for the positive class
        threshold: classification threshold

    Returns:
        dict with keys 'auc', 'fpr', 'fnr', and 'roc_curve' (fpr, tpr arrays)
    """
    auc_val = roc_auc_score(y_true, y_pred_proba)
    y_pred_binary = (np.asarray(y_pred_proba) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)
    return {
        "auc": auc_val,
        "fpr": fpr,
        "fnr": fnr,
        "roc_curve": {"fpr": fpr_curve, "tpr": tpr_curve},
    }


def _fold_aucs(predictions: pd.DataFrame) -> list[float]:
    return [
        auc_or_nan(g.label, g.prob) for _, g in predictions.groupby("fold", sort=True)
    ]


def _summary(fold_auc: list[float]) -> dict:
    valid = [a for a in fold_auc if not np.isnan(a)]
    return {
        "auc": float(np.mean(valid)) if valid else float("nan"),
        # population std across folds, as in the papers
        "auc_std": float(np.std(valid)) if valid else float("nan"),
        "fold_auc": fold_auc,
    }


def cv_report(predictions: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Summarise out-of-fold predictions as in the Pyrocast paper.

    Args:
        predictions: one row per test sample with label, prob, fold and
            state_now (NRL state at the input hour, -1 if missing)
        threshold: classification threshold for the pooled FPR and FNR

    Returns:
        dict with the mean ("auc") and population standard deviation
        ("auc_std") of the per-fold AUCs, the per-fold AUCs, the pooled AUC,
        FPR, FNR and ROC curve, and "auc_by_state": the same fold summary for
        the samples in each NRL state at the input hour (Figure 1 of the paper)
    """
    pooled = classification_report(predictions.label, predictions.prob, threshold)
    report = _summary(_fold_aucs(predictions))
    report.update(
        pooled_auc=pooled["auc"],
        fpr=pooled["fpr"],
        fnr=pooled["fnr"],
        roc_curve=pooled["roc_curve"],
        n_folds=int(predictions.fold.nunique()),
    )
    if "state_now" in predictions:
        states = predictions.state_now.map(NRL_STATE_NAMES)
        report["auc_by_state"] = {
            name: {
                **_summary(_fold_aucs(group)),
                "n": len(group),
                "n_positive": int(group.label.sum()),
            }
            for name, group in predictions.groupby(states, sort=False)
        }
    return report
