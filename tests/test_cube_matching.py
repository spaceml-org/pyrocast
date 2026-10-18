"""Tests for pyrocast.utils.data.cube_matching."""

import os

import numpy as np
import pandas as pd
import zarr

from pyrocast.utils.data.cube_matching import (
    N_HOURS,
    event_flag_arrays,
    event_flags,
    match_samples,
    nrl_states,
)

EVENT_ID = "1_1"


def _write_flags(flag_root, hours):
    """Write 5-element boolean flag arrays; the last element is the PyroCb flag."""
    for hour in hours:
        flag = np.zeros(5, dtype=bool)
        flag[4] = hour % 2 == 0
        path = os.path.join(flag_root, EVENT_ID, f"{hour:02d}_PyroCb_flags.zarr")
        zarr.save_array(path, flag, zarr_format=2)


def _listing(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(root)
        for f in fs
    )


class TestEventFlags:
    def test_flag_files_are_one_based(self, tmp_path):
        _write_flags(tmp_path, range(1, 13))
        flags = event_flags(EVENT_ID, str(tmp_path))
        assert flags.shape == (N_HOURS,)
        np.testing.assert_array_equal(flags[:12], [h % 2 == 0 for h in range(1, 13)])
        assert np.all(flags[12:] == -1)

    def test_flag_arrays(self, tmp_path):
        _write_flags(tmp_path, [1, 2])
        arrays = event_flag_arrays(EVENT_ID, str(tmp_path))
        assert arrays.shape == (N_HOURS, 5)
        np.testing.assert_array_equal(arrays[1], [0, 0, 0, 0, 1])
        assert np.all(arrays[2:] == -1)

    def test_does_not_write_to_flag_root(self, tmp_path):
        _write_flags(tmp_path, range(1, 13))
        before = _listing(tmp_path)
        event_flags(EVENT_ID, str(tmp_path))
        assert _listing(tmp_path) == before


def test_nrl_states():
    arrays = np.array(
        [
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(nrl_states(arrays), [-1, 0, 1, 2, 3, 4])


def _hourly(flags):
    """Hourly table for one event with flags for hours 0, 1, ..."""
    flags = list(flags) + [-1] * (N_HOURS - len(flags))
    return pd.DataFrame(
        {"event_id": EVENT_ID, "date_idx": np.arange(N_HOURS), "flag": flags}
    )


# flags for hours 0-11, missing at hour 3 and from hour 12
FLAGS = [0, 1, 0, -1, 0, 1, 0, 1, 0, 1, 0, 1]


class TestMatchSamples:
    def test_detection(self):
        samples = match_samples(_hourly(FLAGS), "detection")
        assert list(samples.date_idx) == [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        assert list(samples.era5_idx) == list(samples.date_idx)
        assert list(samples.label) == list(samples.flag_now)

    def test_forecast(self):
        samples = match_samples(_hourly(FLAGS), "forecast")
        assert list(samples.date_idx) == [0, 1, 2, 3, 4, 5]
        assert list(samples.era5_idx) == [0, 1, 2, 3, 4, 5]
        assert list(samples.label) == FLAGS[6:12]
        assert list(samples.flag_now) == [0, 1, 0, -1, 0, 1]

    def test_forecast_oracle_reads_era5_at_target(self):
        samples = match_samples(_hourly(FLAGS), "forecast_oracle")
        assert list(samples.date_idx) == [0, 1, 2, 3, 4, 5]
        assert list(samples.era5_idx) == [6, 7, 8, 9, 10, 11]
        assert list(samples.label) == FLAGS[6:12]

    def test_lead_hours(self):
        samples = match_samples(_hourly(FLAGS), "forecast", lead_hours=2)
        assert list(samples.date_idx) == [0, 2, 3, 4, 5, 6, 7, 8, 9]
        assert list(samples.label) == [0, 0, 1, 0, 1, 0, 1, 0, 1]

    def test_events_do_not_mix(self):
        other = _hourly([1] * N_HOURS).assign(event_id="1_2")
        hourly = pd.concat([_hourly(FLAGS[:7]), other], ignore_index=True)
        samples = match_samples(hourly, "forecast")
        assert list(samples.event_id) == ["1_1"] + ["1_2"] * (N_HOURS - 6)


def test_state_now_follows_input_hour():
    hourly = _hourly(FLAGS).assign(nrl_state=np.arange(N_HOURS))
    samples = match_samples(hourly, "forecast")
    assert list(samples.state_now) == list(samples.date_idx)
