"""Tests for pyrocast.utils.metrics."""

import numpy as np
import pytest
from pyrocast.utils.metrics import auc, classification_report


def test_auc_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    assert auc(y_true, y_pred) == 1.0


def test_auc_random():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=200)
    y_pred = rng.rand(200)
    result = auc(y_true, y_pred)
    assert 0.0 <= result <= 1.0


def test_classification_report_keys():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.4, 0.6, 0.9]
    report = classification_report(y_true, y_pred)
    assert set(report.keys()) == {"auc", "fpr", "fnr", "roc_curve"}


def test_classification_report_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    report = classification_report(y_true, y_pred)
    assert report["auc"] == 1.0
    assert report["fpr"] == 0.0
    assert report["fnr"] == 0.0


def test_classification_report_threshold():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.4, 0.6, 0.9]
    report_low = classification_report(y_true, y_pred, threshold=0.3)
    report_high = classification_report(y_true, y_pred, threshold=0.8)
    # low threshold -> more FP, fewer FN
    assert report_low["fnr"] <= report_high["fnr"]
