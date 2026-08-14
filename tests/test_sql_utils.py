"""Tests for pyrocast.utils.sql_utils — pure functions only (no BigQuery)."""

import pytest
from pyrocast.utils.sql_utils import (
    checkdate,
    checkdatetime,
    getsatellite,
    getscanmode,
)


class TestCheckdate:
    def test_future_year(self):
        assert checkdate(2020, 1, 1, 2019, 12, 31) is True

    def test_past_year(self):
        assert checkdate(2018, 1, 1, 2019, 1, 1) is False

    def test_same_date(self):
        assert checkdate(2019, 6, 15, 2019, 6, 15) is True

    def test_same_year_earlier_month(self):
        assert checkdate(2019, 3, 1, 2019, 6, 1) is False


class TestCheckdatetime:
    def test_future_year(self):
        assert checkdatetime(2020, 1, 1, 0, 2019, 12, 31, 23) is True

    def test_same_day_future_hour(self):
        assert checkdatetime(2019, 4, 2, 18, 2019, 4, 2, 16) is True

    def test_same_day_past_hour(self):
        assert checkdatetime(2019, 4, 2, 10, 2019, 4, 2, 16) is False

    def test_past_day(self):
        assert checkdatetime(2019, 4, 1, 23, 2019, 4, 2, 0) is False

    def test_future_day(self):
        assert checkdatetime(2019, 4, 3, 0, 2019, 4, 2, 16) is True


class TestGetsatellite:
    def test_himawari(self):
        assert getsatellite(-33.0, 151.0, "2020-01-01") == "Himawari"

    def test_goes16_east(self):
        assert getsatellite(40.0, -80.0, "2020-01-01") == "GOES16"

    def test_goes17_west(self):
        assert getsatellite(40.0, -120.0, "2020-01-01") == "GOES17"

    def test_goes16_fallback_for_west_before_goes17(self):
        assert getsatellite(40.0, -120.0, "2018-06-01") == "GOES16"

    def test_nan_before_any_satellite(self):
        assert getsatellite(-33.0, 151.0, "2014-01-01") == "NaN"


class TestGetscanmode:
    def test_himawari_returns_none(self):
        from datetime import datetime
        assert getscanmode(datetime(2020, 1, 1, 12), "Himawari") is None

    def test_goes_mode6_after_cutover(self):
        from datetime import datetime
        assert getscanmode(datetime(2020, 1, 1, 12), "GOES16") == 6

    def test_goes_mode3_before_cutover(self):
        from datetime import datetime
        assert getscanmode(datetime(2018, 1, 1, 12), "GOES16") == 3
