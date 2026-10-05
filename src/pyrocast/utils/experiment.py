import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from pyrocast.config import SplitConfig
from pyrocast.utils.data.cross_validation import fold_splits


def _to_builtin(value):
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_run(run_dir: Path, config: BaseModel, metrics: dict) -> None:
    """
    Write the resolved config and metrics of a run to run_dir.

    Args:
        run_dir: output directory, created if missing
        config: validated experiment config, saved as config.yaml
        metrics: JSON-serialisable metrics (numpy values allowed), saved as
            metrics.json
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(_to_builtin(metrics), f, indent=2)


PREDICTION_COLUMNS = [
    "event_id",
    "wildfire_id",
    "date_idx",
    "era5_idx",
    "state_now",
    "label",
]


def cross_validate(
    index: pd.DataFrame,
    split: SplitConfig,
    fit_predict: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> pd.DataFrame:
    """
    Fit and predict on every fold of a split.

    Args:
        index: sample index
        split: split configuration, see cross_validation.fold_splits
        fit_predict: called with the train and test row positions and the fold
            number, returns predicted probabilities for the test rows

    Returns:
        one row per test sample with PREDICTION_COLUMNS, fold and prob
    """
    frames = []
    for fold, (train, test) in enumerate(fold_splits(index, split)):
        logging.info(
            "Fold %d: %d train, %d test samples (%d and %d positive)",
            fold,
            len(train),
            len(test),
            index.label.iloc[train].sum(),
            index.label.iloc[test].sum(),
        )
        prob = fit_predict(train, test, fold)
        frames.append(
            index.iloc[test][PREDICTION_COLUMNS].assign(fold=fold, prob=prob)
        )
    return pd.concat(frames, ignore_index=True)
