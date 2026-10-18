"""Tests for pyrocast.icp.icp."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from sklearn.metrics import roc_auc_score

from pyrocast.icp import icp
from pyrocast.paper import ICP_REFERENCE

VARS11 = ["ch1", "ch6", "u10", "blh", "cape", "z", "sshf", "v", "r850", "uv10", "alt"]


def _delong_reference(y, p0, p1):
    """Textbook O(n^2) DeLong statistic with ties counted as 1/2."""

    def components(p):
        pos, neg = p[y == 1], p[y == 0]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)

    a0, v01_0, v10_0 = components(p0)
    a1, v01_1, v10_1 = components(p1)
    s01 = np.cov(np.stack([v01_0, v01_1]))
    s10 = np.cov(np.stack([v10_0, v10_1]))
    s = s01 / len(v01_0) + s10 / len(v10_0)
    return (a1 - a0) / np.sqrt(s[0, 0] + s[1, 1] - 2 * s[0, 1])


class TestDeLong:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p0 = np.round(y * 0.3 + rng.random(300), 1)  # rounded: many ties
        p1 = np.round(y * 0.6 + rng.random(300), 1)
        return y, p0, p1

    def test_matches_reference(self, data):
        y, p0, p1 = data
        result = icp.delong_test(y, p0, p1)
        assert result["auc_noE"] == pytest.approx(roc_auc_score(y, p0))
        assert result["auc_E"] == pytest.approx(roc_auc_score(y, p1))
        assert result["stat"] == pytest.approx(_delong_reference(y, p0, p1))

    def test_better_model_with_environment(self, data):
        y, p0, p1 = data
        result = icp.delong_test(y, p0, p1)
        assert result["pval_1tail"] < 0.01
        assert result["pval_2tail"] == pytest.approx(2 * result["pval_1tail"])
        swapped = icp.delong_test(y, p1, p0)
        assert swapped["stat"] == pytest.approx(-result["stat"])
        assert swapped["pval_1tail"] > 0.99

    def test_identical_predictions(self, data):
        y, p0, _ = data
        result = icp.delong_test(y, p0, p0)
        assert result["stat"] == 0 and result["pval_1tail"] == 0.5


class FakeTest:
    """Conditional independence test with p = product of per-variable weights."""

    def __init__(self, weights):
        self.weights = weights
        self.calls = []

    def __call__(self, subsets):
        self.calls.append(subsets)
        return [
            {
                "subset": list(s),
                "pval_1tail": float(np.prod([self.weights[v] for v in s])),
                "stat": 0.0,
                "pval_2tail": 1.0,
                "auc_E": 0.5,
                "auc_noE": 0.5,
            }
            for s in subsets
        ]


class TestGreedy:
    def test_order_and_causal_set(self):
        # removing a variable with a small weight raises p the most
        weights = {"a": 0.01, "b": 0.5, "c": 0.9, "d": 0.99}
        steps = icp.greedy_icp(FakeTest(weights), list(weights))
        assert [s["excluded"] for s in steps] == ["a", "b", "c", "d"]
        assert steps[0]["remaining"] == ["b", "c", "d"]
        assert steps[0]["pval_1tail"] == pytest.approx(0.5 * 0.9 * 0.99)
        assert np.isnan(steps[-1]["pval_1tail"])
        # the first test already rejects: nothing is excluded
        assert icp.greedy_causal_set(steps, alpha=0.5) == ["a", "b", "c", "d"]
        # no test rejects: only the last variable is left
        assert icp.greedy_causal_set(steps, alpha=0.05) == ["d"]

    def test_causal_set_from_paper_pvalues(self):
        ref = ICP_REFERENCE
        steps = [
            {"excluded": v, "pval_1tail": np.nan if p is None else p}
            for v, p in zip(ref["greedy_order"], ref["greedy_pvalues"])
        ]
        assert icp.greedy_causal_set(steps, 0.05) == ref["greedy_causal_set"]


def test_subsets_of_size():
    subsets = icp.subsets_of_size(VARS11, 8)
    assert len(subsets) == 232  # as in the paper
    assert all(len(s) >= 8 for s in subsets)


def test_defining_sets():
    accepted = [["a", "b"], ["b", "c"]]
    assert icp.defining_sets(accepted, ["a", "b", "c"]) == [["b"], ["a", "c"]]
    assert icp.defining_sets([], ["a"]) == []


def test_intersection():
    assert icp.intersection([["a", "b", "c"], ["c", "a"]]) == ["a", "c"]
    assert icp.intersection([]) == []


def test_cluster_subsets():
    clusters = [["a", "b"], ["b"], ["c"], ["d", "a"]]
    subsets = icp.cluster_subsets(clusters, ["a", "b", "c", "d"], 3)
    # {a,b}+{b}+{c} and {a,b}+{c}+{d,a} etc.: 5 combinations, 3 distinct unions
    assert subsets == [["a", "b", "c"], ["a", "b", "d"], ["a", "b", "c", "d"]]


def test_hsic_clusters():
    matrix = np.array([[np.nan, 0.3, 0.1], [0.3, np.nan, 0.25], [0.1, 0.25, np.nan]])
    clusters = icp.hsic_clusters(matrix, ["a", "b", "c"], 0.25)
    assert clusters == [["a", "b"], ["b", "a", "c"], ["c", "b"]]


def test_normalised_hsic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    dependent = icp.normalised_hsic(x, x[:, :1])
    independent = icp.normalised_hsic(x, rng.normal(size=(200, 2)))
    assert dependent > 0.2 > independent > 0
    assert icp.normalised_hsic(x, x) == pytest.approx(1.0)


def test_environment():
    index = pd.DataFrame(
        {
            "longitude": [150.0],
            "latitude": [-30.0],
            "datetime": [pd.Timestamp("2020-07-02 05:00")],
        }
    )
    env = icp.environment(index, ["latitude", "date"])
    np.testing.assert_allclose(env, [[-30.0, 2020 + 183 / 366]])


def test_oof_predictions_do_not_use_own_fold():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    y = np.tile([0, 1], 30)
    folds = np.repeat([0, 1, 2], 20)
    cfg = icp.ICPConfig(n_estimators=5)
    p = icp.oof_predictions(x, y, folds, 5, cfg)
    assert p.shape == (60,) and np.all((0 <= p) & (p <= 1))
    # a sample's prediction does not change when its own label changes
    y2 = y.copy()
    y2[folds == 0] = 1 - y2[folds == 0]
    np.testing.assert_array_equal(
        p[folds == 0], icp.oof_predictions(x, y2, folds, 5, cfg)[folds == 0]
    )


@pytest.mark.slow
def test_icp_main(tmp_path, data_root):
    lat, lon = np.arange(89.75, -90, -0.5), np.arange(0.25, 360, 0.5)
    elevation = tmp_path / "elev.nc"
    xr.Dataset(
        {"data": (("lat", "lon"), np.add.outer(lat, lon))},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(elevation)
    path = tmp_path / "icp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "icp",
                "output_dir": str(tmp_path / "out"),
                "data": {
                    "root": str(data_root),
                    "n_jobs": 1,
                    "elevation_path": str(elevation),
                },
                "split": {"scheme": "event_cv", "n_folds": 3},
                "model": {
                    "n_estimators": 3,
                    "validation_n_estimators": 3,
                    "n_jobs": 1,
                    "n_exhaustive": 4,
                    "min_subset_size": 3,
                    "hsic_samples": 20,
                },
            }
        )
    )
    icp.main(["--config", str(path)])
    run_dir = tmp_path / "out" / "icp"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert sorted(metrics["greedy_order"]) == sorted(icp.ICP_VARIABLES)
    assert len(icp.ICP_VARIABLES) == 28
    assert metrics["n_features"] == 26 * 11 + 4 + 6  # 296, as in the paper
    assert metrics["exhaustive"]["variables"] == [
        v for v in icp.ICP_VARIABLES if v in metrics["greedy_order"][-4:]
    ]
    assert metrics["exhaustive"]["n_tested"] == 5
    validation = json.loads((run_dir / "validation.json").read_text())
    assert len(validation) == 2 * 28
    n_tests = len((run_dir / "icp_tests.jsonl").read_text().splitlines())
    # a second run reloads every stage and repeats no test
    icp.main(["--config", str(path)])
    assert len((run_dir / "icp_tests.jsonl").read_text().splitlines()) == n_tests
