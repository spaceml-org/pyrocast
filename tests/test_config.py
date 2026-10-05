"""Tests for pyrocast.config."""

import pytest
import yaml
from pydantic import ValidationError

from pyrocast.config import (
    CNNConfig,
    CNNExperimentConfig,
    DataConfig,
    ICPExperimentConfig,
    RFExperimentConfig,
    RandomForestConfig,
    SplitConfig,
    load_config,
)


class TestDataConfig:
    def test_paths(self, data_root):
        cfg = DataConfig(root=data_root)
        assert cfg.geostationary_path == data_root / "Geostationary_imagery"
        assert cfg.era5_path == data_root / "climate_and_fuel"
        assert cfg.flags_path == data_root / "PyroCb_flags_and_masks"
        assert cfg.snapshots_path == data_root / "wildfire_snapshots.csv"

    def test_missing_root(self, tmp_path):
        with pytest.raises(ValidationError):
            DataConfig(root=tmp_path / "nope")

    def test_missing_store(self, tmp_path):
        with pytest.raises(ValidationError, match="Geostationary_imagery"):
            DataConfig(root=tmp_path)

    def test_missing_events_table(self, data_root):
        with pytest.raises(ValidationError, match="nope.csv"):
            DataConfig(root=data_root, events_csv="nope.csv")

    def test_missing_elevation(self, data_root):
        with pytest.raises(ValidationError):
            DataConfig(root=data_root, elevation_path=data_root / "nope.nc")

    def test_extra_field_forbidden(self, data_root):
        with pytest.raises(ValidationError):
            DataConfig(root=data_root, typo=1)

    @pytest.mark.parametrize("n_jobs", [0, -2])
    def test_invalid_n_jobs(self, data_root, n_jobs):
        with pytest.raises(ValidationError):
            DataConfig(root=data_root, n_jobs=n_jobs)


class TestSplitConfig:
    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
    def test_invalid_fraction(self, fraction):
        with pytest.raises(ValidationError):
            SplitConfig(test_fraction=fraction)

    def test_invalid_scheme(self):
        with pytest.raises(ValidationError):
            SplitConfig(scheme="leave_one_out")

    def test_invalid_n_folds(self):
        with pytest.raises(ValidationError):
            SplitConfig(n_folds=1)


class TestModelConfigs:
    def test_rf_defaults_match_paper(self):
        cfg = RandomForestConfig()
        assert (cfg.n_estimators, cfg.max_depth) == (500, 10)
        assert cfg.class_weight == "balanced_subsample"

    def test_cnn_defaults_match_paper(self):
        cfg = CNNConfig()
        assert (cfg.n_epochs, cfg.batch_size, cfg.lr) == (5, 64, 0.001)
        assert cfg.pretrain_epochs == 0

    def test_cnn_invalid_pretrain_epochs(self):
        with pytest.raises(ValidationError):
            CNNConfig(pretrain_epochs=-1)

    def test_cnn_invalid_device(self):
        with pytest.raises(ValidationError):
            CNNConfig(device="tpu")


class TestLoadConfig:
    def _write(self, tmp_path, content):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(content))
        return path

    def test_rf_experiment(self, tmp_path, data_root):
        path = self._write(
            tmp_path,
            {
                "experiment_name": "rf",
                "output_dir": str(tmp_path / "out"),
                "inputs": "both",
                "data": {"root": str(data_root)},
                "model": {"n_estimators": 10},
            },
        )
        cfg = load_config(path, RFExperimentConfig)
        assert cfg.model.n_estimators == 10
        assert cfg.split.test_fraction == 0.2
        assert cfg.run_dir == tmp_path / "out" / "rf"

    def test_cnn_experiment_rejects_unknown_inputs(self, tmp_path, data_root):
        path = self._write(
            tmp_path,
            {
                "experiment_name": "cnn",
                "output_dir": str(tmp_path),
                "inputs": "radar",
                "data": {"root": str(data_root)},
            },
        )
        with pytest.raises(ValidationError):
            load_config(path, CNNExperimentConfig)


class TestMode:
    def _cfg(self, tmp_path, data_root, **kwargs):
        return RFExperimentConfig(
            experiment_name="rf",
            output_dir=tmp_path,
            data=DataConfig(root=data_root),
            **kwargs,
        )

    def test_default_is_forecast(self, tmp_path, data_root):
        assert self._cfg(tmp_path, data_root).mode == "forecast"

    @pytest.mark.parametrize("inputs", ["era5", "both"])
    def test_oracle_with_era5(self, tmp_path, data_root, inputs):
        cfg = self._cfg(tmp_path, data_root, mode="forecast_oracle", inputs=inputs)
        assert cfg.mode == "forecast_oracle"

    def test_oracle_rejects_geostationary_only(self, tmp_path, data_root):
        with pytest.raises(ValidationError, match="ERA5"):
            self._cfg(
                tmp_path, data_root, mode="forecast_oracle", inputs="geostationary"
            )

    def test_unknown_mode(self, tmp_path, data_root):
        with pytest.raises(ValidationError):
            self._cfg(tmp_path, data_root, mode="nowcast")


class TestEra5Variables:
    def _cfg(self, tmp_path, data_root, **kwargs):
        return RFExperimentConfig(
            experiment_name="rf",
            output_dir=tmp_path,
            data=DataConfig(root=data_root),
            **kwargs,
        )

    def test_channels_in_store_order(self, tmp_path, data_root):
        cfg = self._cfg(
            tmp_path, data_root, inputs="era5", era5_variables=["r650", "cape", "blh"]
        )
        assert cfg.era5_channels == [3, 4, 16]

    def test_all_by_default(self, tmp_path, data_root):
        assert self._cfg(tmp_path, data_root).era5_channels == list(range(19))

    def test_unknown_variable(self, tmp_path, data_root):
        with pytest.raises(ValidationError):
            self._cfg(tmp_path, data_root, inputs="era5", era5_variables=["t2m"])

    def test_needs_era5_inputs(self, tmp_path, data_root):
        with pytest.raises(ValidationError, match="era5_variables"):
            self._cfg(tmp_path, data_root, era5_variables=["cape"])

    def test_duplicates(self, tmp_path, data_root):
        with pytest.raises(ValidationError, match="duplicates"):
            self._cfg(tmp_path, data_root, inputs="era5", era5_variables=["cape"] * 2)


class TestICPExperimentConfig:
    def _cfg(self, tmp_path, data_root, elevation=True, **kwargs):
        elevation_path = tmp_path / "elev.nc"
        elevation_path.touch()
        data = DataConfig(
            root=data_root, elevation_path=elevation_path if elevation else None
        )
        return ICPExperimentConfig(
            experiment_name="icp", output_dir=tmp_path, data=data, **kwargs
        )

    def test_defaults_match_paper(self, tmp_path, data_root):
        cfg = self._cfg(tmp_path, data_root)
        assert cfg.inputs == "both" and cfg.split.scheme == "event_cv"
        assert (cfg.model.n_estimators, cfg.model.alpha) == (100, 0.05)
        assert cfg.model.environment == ["longitude", "latitude", "date"]

    def test_needs_elevation(self, tmp_path, data_root):
        with pytest.raises(ValidationError, match="elevation"):
            self._cfg(tmp_path, data_root, elevation=False)

    def test_rejects_holdout(self, tmp_path, data_root):
        with pytest.raises(ValidationError, match="cross-validation"):
            self._cfg(tmp_path, data_root, split=SplitConfig(scheme="holdout"))

    def test_rejects_subset_of_inputs(self, tmp_path, data_root):
        with pytest.raises(ValidationError, match="all variables"):
            self._cfg(tmp_path, data_root, inputs="era5")
