"""Tests for pyrocast.utils.data.cross_validation."""

import numpy as np
import pandas as pd
import pytest

from pyrocast.config import SplitConfig
from pyrocast.utils.data.cross_validation import (
    fold_splits,
    split_by_fire,
    wildfire_folds,
)


@pytest.fixture
def index():
    """Ten wildfires with three samples each: five near (0, 0), five near (50, 50)."""
    rows = []
    for w in range(10):
        offset = 0.0 if w < 5 else 50.0
        for _ in range(3):
            rows.append(
                {
                    "wildfire_id": f"W{w}",
                    "longitude": offset + w,
                    "latitude": offset - w,
                }
            )
    return pd.DataFrame(rows)


class TestWildfireFolds:
    def test_event_cv_balanced(self, index):
        folds = wildfire_folds(index, SplitConfig(scheme="event_cv", n_folds=5))
        assert sorted(folds.value_counts()) == [2] * 5
        assert list(folds.index) == sorted(index.wildfire_id.unique())

    def test_event_cv_seed(self, index):
        a = wildfire_folds(index, SplitConfig(scheme="event_cv", seed=0))
        b = wildfire_folds(index, SplitConfig(scheme="event_cv", seed=0))
        c = wildfire_folds(index, SplitConfig(scheme="event_cv", seed=1))
        pd.testing.assert_series_equal(a, b)
        assert not a.equals(c)

    def test_spatial_cv_separates_regions(self, index):
        folds = wildfire_folds(index, SplitConfig(scheme="spatial_cv", n_folds=2))
        assert folds.iloc[:5].nunique() == folds.iloc[5:].nunique() == 1
        assert folds.iloc[0] != folds.iloc[5]

    def test_too_few_wildfires(self, index):
        with pytest.raises(ValueError, match="10 wildfires"):
            wildfire_folds(index, SplitConfig(scheme="event_cv", n_folds=11))

    def test_rejects_holdout(self, index):
        with pytest.raises(ValueError):
            wildfire_folds(index, SplitConfig(scheme="holdout"))


class TestFoldSplits:
    @pytest.mark.parametrize("scheme", ["event_cv", "spatial_cv"])
    def test_partition_by_wildfire(self, index, scheme):
        splits = fold_splits(index, SplitConfig(scheme=scheme, n_folds=5))
        assert len(splits) == 5
        tested = np.concatenate([test for _, test in splits])
        assert sorted(tested) == list(range(len(index)))
        for train, test in splits:
            assert set(index.wildfire_id.iloc[train]).isdisjoint(
                index.wildfire_id.iloc[test]
            )
            assert len(train) + len(test) == len(index)

    def test_holdout(self, index):
        [(train, test)] = fold_splits(index, SplitConfig(test_fraction=0.2))
        assert index.wildfire_id.iloc[test].nunique() == 2
        assert set(index.wildfire_id.iloc[train]).isdisjoint(
            index.wildfire_id.iloc[test]
        )


class TestSplitByFire:
    def test_no_wildfire_in_both_splits(self, index):
        train, test = split_by_fire(index, SplitConfig(test_fraction=0.3, seed=0))
        assert set(train.wildfire_id).isdisjoint(test.wildfire_id)
        assert len(train) + len(test) == len(index)

    def test_ignores_cv_scheme(self, index):
        split = SplitConfig(scheme="event_cv", test_fraction=0.3, seed=3)
        a, _ = split_by_fire(index, split)
        b, _ = split_by_fire(index, split.model_copy(update={"scheme": "holdout"}))
        pd.testing.assert_frame_equal(a, b)
