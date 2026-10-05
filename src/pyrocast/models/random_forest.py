import argparse
import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pyrocast.config import RFExperimentConfig, load_config
from pyrocast.utils import metrics
from pyrocast.utils.data.dataload import get_sample_index
from pyrocast.utils.data.features import (
    get_feature_table,
    input_variables,
    sample_features,
)
from pyrocast.utils.experiment import cross_validate, save_run


class RandomForest(RandomForestClassifier):
    """SciKit Learn model initalised with paper hyperparameters"""

    def __init__(
        self,
        n_estimators=500,
        max_depth=10,
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )


def run(cfg: RFExperimentConfig) -> dict:
    """
    Train and evaluate a random forest on the 11 summary features of each input
    variable, saving the config, metrics and out-of-fold predictions.

    Args:
        cfg: random forest experiment config

    Returns:
        metrics from pyrocast.utils.metrics.cv_report plus sample counts and
        the impurity importance of each input variable (summed over its
        features, averaged over folds)
    """
    index = get_sample_index(cfg.data, cfg.mode)
    logging.info(
        "Sample index (%s): %d samples, %d events, %d wildfires",
        cfg.mode,
        len(index),
        index.event_id.nunique(),
        index.wildfire_id.nunique(),
    )
    variables = input_variables(cfg.inputs, cfg.era5_variables)
    x, _ = sample_features(index, get_feature_table(cfg.data, index), variables)
    y = index.label.to_numpy(dtype=int)

    models = []

    def fit_predict(train, test, fold):
        rf = RandomForest(**cfg.model.model_dump())
        rf.fit(x[train], y[train])
        models.append(rf)
        return rf.predict_proba(x[test])[:, 1]

    predictions = cross_validate(index, cfg.split, fit_predict)
    importance = np.mean([m.feature_importances_ for m in models], axis=0)
    results = metrics.cv_report(predictions)
    results.update(
        n_samples=len(index),
        n_positive=int(y.sum()),
        n_events=int(index.event_id.nunique()),
        n_wildfires=int(index.wildfire_id.nunique()),
        n_features=x.shape[1],
        importance=dict(
            zip(variables, importance.reshape(len(variables), -1).sum(axis=1))
        ),
    )
    save_run(cfg.run_dir, cfg, results)
    predictions.to_csv(cfg.run_dir / "predictions.csv", index=False)
    if cfg.split.scheme == "holdout":
        joblib.dump(models[0], cfg.run_dir / "model.joblib")
    logging.info("AUC: %.3f +/- %.3f", results["auc"], results["auc_std"])
    return results


def main(argv: list[str] | None = None) -> None:
    """
    Train a random forest from a YAML config.

    Args:
        argv: command-line arguments, defaults to sys.argv
    """
    parser = argparse.ArgumentParser(description="Train the Pyrocast random forest")
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run(load_config(args.config, RFExperimentConfig))


if __name__ == "__main__":
    main()
