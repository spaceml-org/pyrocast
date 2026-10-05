"""Tests for the config grids and command line in pyrocast.utils.experiment."""

import json

import pytest
import yaml
from pydantic import ValidationError

from pyrocast.config import RFExperimentConfig
from pyrocast.models import random_forest
from pyrocast.utils.experiment import expand_grid, load_experiments

BASE = {"experiment_name": "exp", "output_dir": "out", "split": {"seed": 0}}


def test_no_grid():
    assert expand_grid(BASE) == [BASE]


def test_list_values():
    runs = expand_grid({**BASE, "grid": {"mode": ["detection", "forecast"]}})
    assert [r["mode"] for r in runs] == ["detection", "forecast"]
    assert [r["experiment_name"] for r in runs] == ["exp/detection", "exp/forecast"]
    assert all("grid" not in r for r in runs)


def test_named_values_and_dotted_keys():
    grid = {
        "split.scheme": {"event": "event_cv", "spatial": "spatial_cv"},
        "inputs": {
            "gs": {"inputs": "geostationary"},
            "w3": {"inputs": "era5", "era5_variables": ["cape"]},
        },
    }
    runs = expand_grid({**BASE, "grid": grid})
    assert [r["experiment_name"] for r in runs] == [
        "exp/event_gs",
        "exp/event_w3",
        "exp/spatial_gs",
        "exp/spatial_w3",
    ]
    assert runs[3]["split"] == {"seed": 0, "scheme": "spatial_cv"}
    assert runs[3]["era5_variables"] == ["cape"]
    assert "era5_variables" not in runs[0]
    assert BASE["split"] == {"seed": 0}  # not mutated


def test_exclude():
    raw = {
        **BASE,
        "grid": {"mode": ["detection", "forecast"], "n": [1, 2]},
        "exclude": [{"mode": "forecast", "n": 1}],
    }
    names = [r["experiment_name"] for r in expand_grid(raw)]
    assert names == ["exp/detection_1", "exp/detection_2", "exp/forecast_2"]


@pytest.mark.parametrize(
    "raw",
    [
        {**BASE, "grid": {"mode": "detection"}},
        {**BASE, "grid": {"mode": ["detection"]}, "exclude": [{"inputs": "gs"}]},
        {**BASE, "exclude": [{"mode": "detection"}]},
    ],
)
def test_invalid_grid(raw):
    with pytest.raises(ValueError):
        expand_grid(raw)


def _write(tmp_path, data_root, **extra):
    path = tmp_path / "config.yaml"
    config = {
        "experiment_name": "sweep",
        "output_dir": str(tmp_path / "out"),
        "inputs": "era5",
        "era5_variables": ["cape"],
        "data": {"root": str(data_root), "n_jobs": 1},
        "split": {"test_fraction": 0.34},
        "model": {"n_estimators": 3},
        **extra,
    }
    path.write_text(yaml.safe_dump(config))
    return path


def test_invalid_run_is_reported(tmp_path, data_root):
    path = _write(
        tmp_path, data_root, grid={"inputs": ["era5", "geostationary"]}
    )  # era5_variables needs ERA5 inputs
    with pytest.raises(ValidationError, match="era5_variables"):
        load_experiments(path, RFExperimentConfig)


def test_main_runs_grid(tmp_path, data_root, capsys):
    path = _write(tmp_path, data_root, grid={"mode": ["detection", "forecast"]})
    random_forest.main(["--config", str(path), "--list"])
    listed = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in listed] == ["0", "1"]
    random_forest.main(["--config", str(path), "--index", "1"])
    out = tmp_path / "out" / "sweep"
    assert (out / "forecast" / "metrics.json").exists()
    assert not (out / "detection").exists()
    random_forest.main(["--config", str(path)])
    metrics = json.loads((out / "detection" / "metrics.json").read_text())
    assert metrics["n_samples"] == 72
    with pytest.raises(SystemExit):
        random_forest.main(["--config", str(path), "--index", "2"])
