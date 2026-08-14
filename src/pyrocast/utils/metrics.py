from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np


def auc(ytest, ypred):
    return roc_auc_score(ytest, ypred)


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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)
    return {
        "auc": auc_val,
        "fpr": fpr,
        "fnr": fnr,
        "roc_curve": {"fpr": fpr_curve, "tpr": tpr_curve},
    }
