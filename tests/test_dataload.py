"""Tests for the sample index and cube readers in pyrocast.utils.data.dataload."""

import numpy as np
import pandas as pd
import pytest

from conftest import NO_ERA5_EVENT, SIX_CHANNEL_EVENT, VALID_EVENTS, listing, nrl_flag
from pyrocast.config import DataConfig
from pyrocast.utils.data.dataload import (
    GEO_CHANNELS,
    HOURLY_COLUMNS,
    build_sample_index,
    get_sample_index,
    read_cubes,
    read_wildfires,
)


@pytest.fixture
def data_cfg(data_root):
    return DataConfig(root=data_root, n_jobs=1)


class TestBuildSampleIndex:
    def test_only_valid_events(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast")
        assert set(index.event_id) == set(VALID_EVENTS)
        assert SIX_CHANNEL_EVENT not in set(index.event_id)
        assert NO_ERA5_EVENT not in set(index.event_id)

    def test_columns_and_values(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast")
        event = index[index.event_id == "1_1"]
        assert list(event.date_idx) == [0, 1, 2, 3, 4, 5]
        assert list(event.era5_idx) == [0, 1, 2, 3, 4, 5]
        assert list(event.label) == [0, 1, 0, 1, 0, 1]
        assert list(event.flag_now) == [0, 1, 0, 1, 0, 1]
        assert set(event.satellite) == {"GOES16"}
        assert set(event.fire_id) == {1}
        assert event.datetime.iloc[0] == pd.Timestamp("2020-01-01 18:00:00")
        assert event.datetime.iloc[5] == pd.Timestamp("2020-01-01 23:00:00")

    def test_nrl_state(self, data_cfg):
        index = build_sample_index(data_cfg, "detection")
        event = index[index.event_id == "1_1"]
        expected = [4 if h % 2 == 0 else 2 if h % 3 == 0 else 1 for h in range(1, 13)]
        assert list(event.state_now) == expected

    def test_wildfire_and_country(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast")
        by_fire = index.groupby("fire_id")[["wildfire_id", "country"]].first()
        assert by_fire.loc[1].tolist() == ["W1", "Australia"]
        assert by_fire.loc[2].tolist() == ["pyrocb_2", "US"]

    def test_countries(self, data_root):
        cfg = DataConfig(root=data_root, n_jobs=1, countries=["US", "Canada"])
        index = build_sample_index(cfg, "forecast")
        assert set(index.fire_id) == {2, 3}

    def test_countries_before_max_events(self, data_root):
        cfg = DataConfig(root=data_root, n_jobs=1, countries=["Canada"], max_events=1)
        assert set(build_sample_index(cfg, "forecast").event_id) == {"3_1"}

    def test_detection(self, data_cfg):
        index = build_sample_index(data_cfg, "detection")
        event = index[index.event_id == "1_1"]
        assert list(event.date_idx) == list(range(12))
        assert list(event.label) == [int(h % 2 == 0) for h in range(1, 13)]

    def test_forecast_oracle(self, data_cfg):
        index = build_sample_index(data_cfg, "forecast_oracle")
        event = index[index.event_id == "1_1"]
        assert list(event.date_idx) == [0, 1, 2, 3, 4, 5]
        assert list(event.era5_idx) == [6, 7, 8, 9, 10, 11]

    def test_max_events(self, data_root):
        index = build_sample_index(
            DataConfig(root=data_root, n_jobs=1, max_events=2), "forecast"
        )
        assert index.event_id.nunique() == 2

    def test_does_not_write_to_data_root(self, data_cfg, data_root):
        before = listing(data_root)
        build_sample_index(data_cfg, "forecast")
        assert listing(data_root) == before


class TestGetSampleIndex:
    def test_cache_round_trip(self, data_root, tmp_path):
        cache = tmp_path / "index.csv"
        cfg = DataConfig(root=data_root, n_jobs=1, index_cache=cache)
        built = get_sample_index(cfg, "forecast")
        assert cache.exists()
        loaded = get_sample_index(cfg, "forecast")
        pd.testing.assert_frame_equal(built, loaded)

    def test_cache_shared_between_modes(self, data_root, tmp_path):
        cache = tmp_path / "index.csv"
        cfg = DataConfig(root=data_root, n_jobs=1, index_cache=cache)
        get_sample_index(cfg, "forecast")
        mtime = cache.stat().st_mtime_ns
        detection = get_sample_index(cfg, "detection")
        assert cache.stat().st_mtime_ns == mtime
        expected = build_sample_index(cfg, "detection")
        pd.testing.assert_frame_equal(detection, expected)

    def test_outdated_cache_is_rebuilt(self, data_root, tmp_path):
        cache = tmp_path / "index.csv"
        pd.DataFrame({"event_id": ["1_1"], "flag_then": [1]}).to_csv(cache)
        cfg = DataConfig(root=data_root, n_jobs=1, index_cache=cache)
        assert get_sample_index(cfg, "forecast").event_id.nunique() == len(VALID_EVENTS)
        assert list(pd.read_csv(cache).columns) == HOURLY_COLUMNS

    def test_cache_holds_all_countries(self, data_root, tmp_path):
        cache = tmp_path / "index.csv"
        cfg = DataConfig(
            root=data_root, n_jobs=1, index_cache=cache, countries=["Australia"]
        )
        assert set(get_sample_index(cfg, "forecast").fire_id) == {1}
        assert pd.read_csv(cache).event_id.nunique() == len(VALID_EVENTS)

    def test_cache_holds_all_events_when_max_events_set(self, data_root, tmp_path):
        cache = tmp_path / "index.csv"
        cfg = DataConfig(root=data_root, n_jobs=1, index_cache=cache, max_events=3)
        assert get_sample_index(cfg, "forecast").event_id.nunique() == 3
        assert get_sample_index(cfg, "forecast").event_id.nunique() == 3
        full = DataConfig(root=data_root, n_jobs=1, index_cache=cache)
        assert get_sample_index(full, "forecast").event_id.nunique() == len(
            VALID_EVENTS
        )


class TestReadCubes:
    def test_geostationary_channels_by_satellite(self, data_cfg):
        goes = read_cubes(data_cfg, "1_1", [0, 2], "GOES16", "geostationary")
        assert goes.shape == (2, 6, 200, 200)
        np.testing.assert_array_equal(goes[0, :, 0, 0], GEO_CHANNELS["GOES16"])
        np.testing.assert_array_equal(
            goes[1, :, 0, 0], np.array(GEO_CHANNELS["GOES16"]) + 200
        )
        himawari = read_cubes(data_cfg, "2_1", [0], "Himawari", "geostationary")
        offset = 2
        np.testing.assert_array_equal(
            himawari[0, :, 0, 0], np.array(GEO_CHANNELS["Himawari"]) + offset
        )

    def test_era5(self, data_cfg):
        cube = read_cubes(data_cfg, "1_1", [1], "GOES16", "era5")
        assert cube.shape == (1, 19, 200, 200)
        np.testing.assert_array_equal(cube[0, :, 0, 0], np.arange(19) + 100)

    def test_era5_hours(self, data_cfg):
        cube = read_cubes(data_cfg, "1_1", [0], "GOES16", "both", era5_idxs=[2])
        np.testing.assert_array_equal(cube[0, :6, 0, 0], GEO_CHANNELS["GOES16"])
        np.testing.assert_array_equal(cube[0, 6:, 0, 0], np.arange(19) + 200)

    def test_era5_channels(self, data_cfg):
        cube = read_cubes(data_cfg, "1_1", [1], "GOES16", "both", era5_channels=[4, 3])
        assert cube.shape == (1, 8, 200, 200)
        np.testing.assert_array_equal(cube[0, 6:, 0, 0], [104, 103])

    def test_both(self, data_cfg):
        cube = read_cubes(data_cfg, "1_1", [0], "GOES16", "both")
        assert cube.shape == (1, 25, 200, 200)
        assert cube.dtype == np.float32

    def test_unknown_satellite(self, data_cfg):
        with pytest.raises(KeyError):
            read_cubes(data_cfg, "1_1", [0], "Meteosat", "geostationary")


def test_read_wildfires(data_cfg):
    wildfires = read_wildfires(data_cfg.events_path)
    assert wildfires.loc[1, "wildfire_id"] == "W1"
    assert wildfires.loc[2, "wildfire_id"] == "pyrocb_2"
    assert wildfires.loc[4, "country"] == "Russia"
