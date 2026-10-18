"""Tests for pyrocast.utils.data.features."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import listing
from pyrocast.config import DataConfig
from pyrocast.utils.data.dataload import build_sample_index
from pyrocast.utils.data.features import (
    N_STATS,
    N_TYPE_CODES,
    STATS,
    TABLE_COLUMNS,
    altitude_features,
    altitude_field,
    get_feature_table,
    input_variables,
    load_elevation,
    sample_features,
    stat_columns,
    summary_features,
    type_fractions,
)

ORDER = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2"]


def _offsets(index):
    return index.event_id.map(ORDER.index).to_numpy()


@pytest.fixture
def data_cfg(data_root):
    return DataConfig(root=data_root, n_jobs=1)


def test_summary_features():
    cubes = np.zeros((2, 2, 10, 10), dtype=np.float32)
    cubes[:, 1] = np.arange(100).reshape(10, 10)
    feats = summary_features(cubes)
    assert feats.shape == (2, 2 * N_STATS) and len(STATS) == N_STATS == 11
    np.testing.assert_array_equal(feats[0, :N_STATS], 0)
    second = dict(zip(STATS, feats[0, N_STATS:]))
    assert second["mean"] == 49.5 and second["min"] == 0 and second["max"] == 99
    assert second["p50"] == 49.5
    assert second["std"] == pytest.approx(np.arange(100).std())


def test_type_fractions():
    fields = np.array([[[0.0, 3.2], [2.8, 19.6]]])
    fractions = type_fractions(fields)
    assert fractions.shape == (1, N_TYPE_CODES)
    assert fractions[0, 0] == fractions[0, 20] == 0.25
    assert fractions[0, 3] == 0.5


class TestFeatureTable:
    def test_values(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast_oracle")
        table = get_feature_table(data_cfg, index)
        assert list(table.columns) == TABLE_COLUMNS
        # geostationary hours 0-5 and ERA5 hours 6-11 of each event
        assert len(table) == 12 * len(ORDER)
        row = table.loc[("1_1", 2)]
        assert row["ch1__mean"] == 200 + 0  # GOES ch1 is store channel 0
        assert row["ch4__p99"] == 200 + 6
        assert row["cape__max"] == 200 + 4
        assert row["uv10__mean"] == pytest.approx(np.hypot(200, 201))
        # tvh (store channel 14) holds 214: clipped to the last type code
        assert row["typeH__c20"] == 1.0

    def test_sample_features_use_era5_idx(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast_oracle")
        table = get_feature_table(data_cfg, index)
        x, names = sample_features(index, table, ["ch1", "cape"])
        assert names == stat_columns("ch1") + stat_columns("cape")
        offsets = _offsets(index)
        np.testing.assert_array_equal(x[:, 0], index.date_idx * 100 + offsets)
        np.testing.assert_array_equal(
            x[:, N_STATS], (index.date_idx + 6) * 100 + 4 + offsets
        )

    def test_unknown_variable(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast")
        with pytest.raises(KeyError, match="t2m"):
            sample_features(index, get_feature_table(data_cfg, index), ["t2m"])

    def test_cache_is_incremental(self, data_root, tmp_path):
        cache = tmp_path / "features.pkl"
        cfg = DataConfig(root=data_root, n_jobs=1, feature_cache=cache)
        forecast = build_sample_index(cfg, "forecast")
        first = get_feature_table(cfg, forecast)
        assert len(pd.read_pickle(cache)) == len(first) == 6 * len(ORDER)
        detection = build_sample_index(cfg, "detection")
        second = get_feature_table(cfg, detection)
        assert len(second) == 12 * len(ORDER)
        pd.testing.assert_frame_equal(
            second.loc[first.index].sort_index(), first.sort_index()
        )

    def test_outdated_cache_is_rebuilt(self, data_root, tmp_path):
        cache = tmp_path / "features.pkl"
        pd.DataFrame({"event_id": ["1_1"], "date_idx": [0], "x": [1.0]}).to_pickle(
            cache
        )
        cfg = DataConfig(root=data_root, n_jobs=1, feature_cache=cache)
        index = build_sample_index(cfg, "forecast")
        assert list(get_feature_table(cfg, index).columns) == TABLE_COLUMNS

    def test_does_not_write_to_data_root(self, data_cfg, data_root):
        before = listing(data_root)
        get_feature_table(data_cfg, build_sample_index(data_cfg, "forecast"))
        assert listing(data_root) == before


def test_input_variables():
    assert input_variables("geostationary", None) == [f"ch{i}" for i in range(1, 7)]
    assert input_variables("era5", ["r650", "cape", "blh"]) == ["blh", "cape", "r650"]
    assert len(input_variables("both", None)) == 25


@pytest.fixture
def elevation_file(tmp_path):
    """Elevation equal to 10 * latitude + longitude (0-360) on a 0.5 degree grid."""
    lat = np.arange(89.75, -90, -0.5)
    lon = np.arange(0.25, 360, 0.5)
    data = 10 * lat[:, None] + lon[None, :]
    path = tmp_path / "elev.nc"
    xr.Dataset(
        {"data": (("time", "lat", "lon"), data[None])},
        coords={"time": [0.0], "lat": lat, "lon": lon},
    ).to_netcdf(path)
    return path


def test_altitude_field(elevation_file):
    elevation = load_elevation(elevation_file)
    field = altitude_field(elevation, longitude=100.0, latitude=-30.0, size=200)
    assert field.shape == (200, 200)
    assert field.mean() == pytest.approx(10 * -30 + 100, abs=0.01)
    # 200 km spans about 1.8 degrees of latitude
    assert field[-1, 0] - field[0, 0] == pytest.approx(10 * 199 / 111.32, rel=1e-3)
    # negative longitudes wrap to 0-360
    west = altitude_field(elevation, longitude=-100.0, latitude=0.0, size=2)
    assert west.mean() == pytest.approx(260, abs=0.01)


def test_altitude_features(elevation_file):
    index = pd.DataFrame(
        {"event_id": ["a", "b", "a"], "longitude": [100, 120, 100], "latitude": 0.0}
    )
    feats = altitude_features(index, elevation_file, size=20)
    assert feats.shape == (3, N_STATS)
    np.testing.assert_array_equal(feats[0], feats[2])
    assert feats[1, 0] - feats[0, 0] == pytest.approx(20, abs=0.01)
