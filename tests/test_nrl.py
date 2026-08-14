"""Tests for pyrocast.nrl_algorithm.pyroconvection_detection."""

import numpy as np
import pytest
from pyrocast.nrl_algorithm.pyroconvection_detection import (
    pyro_detection,
    get_full_flag,
    datetime_to_sza,
)
from datetime import datetime, timezone


class TestGetFullFlag:
    def test_no_flags(self):
        assert get_full_flag(np.array([False, False, False])) == 0

    def test_last_flag_set(self):
        assert get_full_flag(np.array([False, False, True])) == 2

    def test_first_flag_set(self):
        assert get_full_flag(np.array([True, False, False])) == 0

    def test_all_flags_set(self):
        assert get_full_flag(np.array([True, True, True])) == 2

    def test_five_element_flag(self):
        flags = np.array([True, True, True, True, True])
        assert get_full_flag(flags) == 4


class TestDatetimeToSza:
    def test_midday_low_sza(self):
        dt = datetime(2020, 6, 21, 12, 0, 0)
        sza = datetime_to_sza(0.0, 0.0, dt)
        assert 0 < sza < 90

    def test_midnight_high_sza(self):
        dt = datetime(2020, 6, 21, 0, 0, 0)
        sza = datetime_to_sza(0.0, 0.0, dt)
        assert sza > 80


class TestPyroDetection:
    def test_nighttime_no_detection(self):
        # midnight -> SZA > 80 -> no detection
        cubes = np.zeros((1, 6, 200, 200))
        dt_list = [datetime(2020, 6, 21, 0, 0, 0)]
        flags, masks = pyro_detection(cubes, dt_list, 0.0, 0.0)
        assert flags[0, 0] is np.True_  # file exists
        assert flags[0, 1] is np.False_  # daytime test fails

    def test_daytime_no_deep_convection(self):
        # warm scene -> no deep convection
        cubes = np.full((1, 6, 200, 200), 300.0)
        dt_list = [datetime(2020, 6, 21, 12, 0, 0)]
        flags, masks = pyro_detection(cubes, dt_list, 0.0, 0.0)
        assert flags[0, 1] is np.True_  # daytime passes
        assert flags[0, 2] is np.False_  # deep convection fails
