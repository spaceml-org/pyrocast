import argparse
import copy
import itertools
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
        frames.append(index.iloc[test][PREDICTION_COLUMNS].assign(fold=fold, prob=prob))
    return pd.concat(frames, ignore_index=True)


GRID_KEY = "grid"
EXCLUDE_KEY = "exclude"


def _set(config: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = copy.deepcopy(value)


def _variants(key: str, values) -> list[tuple[str, dict]]:
    """(label, overrides) of each value of a grid entry."""
    if isinstance(values, dict):
        return [
            (str(name), value if isinstance(value, dict) else {key: value})
            for name, value in values.items()
        ]
    if isinstance(values, list):
        return [(str(v), {key: v}) for v in values]
    raise ValueError(f"grid entry {key!r} must be a list or a mapping")


def expand_grid(raw: dict) -> list[dict]:
    """
    Expand a config with a grid into one config per combination of grid values.

    A config without a "grid" section is returned unchanged. Each grid entry maps a
    dotted config key (e.g. "split.scheme") to either a list of values, labelled by
    their string form, or a mapping from labels to values. In a mapping, a dict
    value is a set of dotted overrides from the config root, so one label can set
    several keys (e.g. inputs and era5_variables). Each run's experiment_name is
    "<experiment_name>/<labels joined by _>", in grid order. "exclude" lists
    partial combinations of labels, as {grid key: label}, to skip.

    Args:
        raw: config dict as loaded from YAML

    Returns:
        one config dict per run, without grid and exclude
    """
    raw = copy.deepcopy(raw)
    grid = raw.pop(GRID_KEY, None)
    exclude = raw.pop(EXCLUDE_KEY, [])
    if not grid:
        if exclude:
            raise ValueError("exclude needs a grid")
        return [raw]
    keys = list(grid)
    for rule in exclude:
        unknown = set(rule) - set(keys)
        if unknown:
            raise ValueError(f"exclude refers to keys not in the grid: {unknown}")
    runs = []
    for combo in itertools.product(*(_variants(k, grid[k]) for k in keys)):
        labels = dict(zip(keys, (label for label, _ in combo)))
        if any(all(labels[k] == str(v) for k, v in r.items()) for r in exclude):
            continue
        config = copy.deepcopy(raw)
        for _, overrides in combo:
            for dotted, value in overrides.items():
                _set(config, dotted, value)
        name = "_".join(labels.values())
        config["experiment_name"] = f"{raw['experiment_name']}/{name}"
        runs.append(config)
    return runs


def load_experiments(path: str | Path, config_cls: type[BaseModel]) -> list:
    """
    Load and validate every run of a YAML config, see expand_grid.

    Args:
        path: path to the YAML file
        config_cls: config class to validate each run against

    Returns:
        validated configs, in grid order
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [config_cls.model_validate(run) for run in expand_grid(raw)]


def experiment_main(
    argv: list[str] | None,
    config_cls: type[BaseModel],
    run: Callable,
    description: str,
) -> None:
    """
    Command line of the training entry points: run every run of a config, or one.

    Args:
        argv: command-line arguments, defaults to sys.argv
        config_cls: config class of the runs
        run: function training one validated config
        description: help text
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument(
        "--index",
        type=int,
        help="only run the run with this index (0-based, e.g. a slurm array task)",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the runs of the config and exit"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    configs = load_experiments(args.config, config_cls)
    if args.list:
        for i, cfg in enumerate(configs):
            print(i, cfg.run_dir)
        return
    if args.index is not None:
        if not 0 <= args.index < len(configs):
            parser.error(f"--index must be in 0..{len(configs) - 1}")
        configs = [configs[args.index]]
    for cfg in configs:
        logging.info("Run %s", cfg.run_dir)
        run(cfg)
