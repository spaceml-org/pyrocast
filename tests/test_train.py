"""End-to-end tests for the random forest and CNN training entry points."""

import json

import joblib
import pandas as pd
import pytest
import torch
import yaml

from pyrocast.models import cnn, random_forest


def _write_config(tmp_path, data_root, inputs, model, mode="forecast", **extra):
    path = tmp_path / "config.yaml"
    config = {
        "experiment_name": "test_run",
        "output_dir": str(tmp_path / "out"),
        "mode": mode,
        "inputs": inputs,
        "data": {"root": str(data_root), "n_jobs": 1},
        "split": {"test_fraction": 0.34, "seed": 0},
        "model": model,
    }
    path.write_text(yaml.safe_dump({**config, **extra}))
    return path


@pytest.mark.parametrize(
    "mode, n_samples",
    [("detection", 72), ("forecast", 36), ("forecast_oracle", 36)],
)
def test_random_forest_main(tmp_path, data_root, mode, n_samples):
    path = _write_config(tmp_path, data_root, "both", {"n_estimators": 5}, mode)
    assert random_forest.main(["--config", str(path)]) is None
    run_dir = tmp_path / "out" / "test_run"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["n_samples"] == n_samples and metrics["n_folds"] == 1
    assert metrics["n_features"] == 25 * 11
    assert set(metrics["importance"]) == {f"ch{i}" for i in range(1, 7)} | {
        "u10",
        "v10",
        "fg10",
        "blh",
        "cape",
        "cin",
        "z",
        "slhf",
        "sshf",
        "w",
        "u",
        "v",
        "cvh",
        "cvl",
        "tvh",
        "tvl",
        "r650",
        "r750",
        "r850",
    }
    assert joblib.load(run_dir / "model.joblib").n_estimators == 5
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert (saved["inputs"], saved["mode"]) == ("both", mode)
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert predictions.prob.between(0, 1).all()


@pytest.mark.parametrize("scheme", ["event_cv", "spatial_cv"])
def test_random_forest_cross_validation(tmp_path, data_root, scheme):
    path = _write_config(
        tmp_path,
        data_root,
        "era5",
        {"n_estimators": 5},
        "forecast_oracle",
        era5_variables=["cape", "blh", "r650"],
        split={"scheme": scheme, "n_folds": 3},
    )
    random_forest.main(["--config", str(path)])
    run_dir = tmp_path / "out" / "test_run"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_folds"] == 3 and len(metrics["fold_auc"]) == 3
    assert metrics["n_features"] == 3 * 11
    assert set(metrics["importance"]) == {"cape", "blh", "r650"}
    assert set(metrics["auc_by_state"]) <= {
        "none",
        "convection",
        "deep_convection",
        "pyrocb",
    }
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == metrics["n_samples"] == 36
    assert predictions.groupby("wildfire_id").fold.nunique().max() == 1
    assert not (run_dir / "model.joblib").exists()


@pytest.mark.slow
@pytest.mark.parametrize("pretrain_epochs", [0, 1])
def test_cnn_main(tmp_path, data_root, pretrain_epochs):
    model = {
        "n_epochs": 1,
        "pretrain_epochs": pretrain_epochs,
        "batch_size": 4,
        "num_workers": 0,
        "norm_samples": 8,
        "device": "cpu",
    }
    path = _write_config(
        tmp_path, data_root, "both", model, era5_variables=["cape", "blh", "r650"]
    )
    assert cnn.main(["--config", str(path)]) is None
    run_dir = tmp_path / "out" / "test_run"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert 0.0 <= metrics["auc"] <= 1.0
    assert len(metrics["losses"]) == 1 and len(metrics["losses"][0]) == 1
    assert ("pretrain_losses" in metrics) == bool(pretrain_epochs)
    state = torch.load(run_dir / "model_fold0.pt")
    assert state["encoder.convs.0.weight"].shape[1] == 9
    assert pd.read_csv(run_dir / "predictions.csv").prob.between(0, 1).all()


def test_cnn_shapes():
    model = cnn.CNN(7)
    x = torch.randn(2, 7, 200, 200)
    assert model(x).shape == (2, 2)
    assert cnn.AutoEncoder(model.encoder, 7)(x).shape == x.shape


def test_resolve_device():
    assert cnn.resolve_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert cnn.resolve_device("auto").type == expected
