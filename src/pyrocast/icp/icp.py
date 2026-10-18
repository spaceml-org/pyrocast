"""
Invariant causal prediction (ICP) of pyroCb drivers.

Implements "Identifying the Causes of Pyrocumulonimbus (PyroCb)" (Díaz
Salas-Porras et al., NeurIPS 2022 Causal ML workshop, arXiv:2211.08883):

- the conditional independence test Y _||_ E | X_S compares the out-of-fold AUC
  of random forests on X_S and on (X_S, E) with DeLong's test, where E is the
  wildfire longitude, latitude and date and folds group observations by wildfire;
- greedy ICP removes, one at a time, the variable whose removal gives the
  largest p-value;
- exhaustive ICP tests every large subset of the variables left by greedy ICP;
- ICP on clusters groups dependent variables by normalised HSIC (Appendix A.2);
- the greedy ordering is validated with event and spatial cross-validation
  (Figure 3).

Each of the 28 candidate variables is summarised by 11 spatial statistics, except
the categorical vegetation types, summarised by the fraction of pixels of their
most common type codes.
"""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from sklearn.ensemble import RandomForestClassifier

from pyrocast.config import (
    ERA5_VARIABLES,
    ICPConfig,
    ICPExperimentConfig,
    SplitConfig,
)
from pyrocast.utils.data.cross_validation import wildfire_folds
from pyrocast.utils.data.dataload import get_sample_index
from pyrocast.utils.data.features import (
    ALTITUDE,
    GEO_VARIABLES,
    VEGETATION_TYPES,
    WIND_SPEEDS,
    altitude_features,
    get_feature_table,
    sample_features,
)
from pyrocast.utils.experiment import experiment_main, save_run
from pyrocast.utils.metrics import auc_or_nan

# Candidate causes of the ICP paper (Table 1 plus wind speeds and altitude). The
# vegetation types enter as typeH and typeL, not as summary statistics.
ICP_VARIABLES = (
    list(GEO_VARIABLES)
    + [v for v in ERA5_VARIABLES if v not in VEGETATION_TYPES.values()]
    + list(WIND_SPEEDS)
    + [ALTITUDE]
    + list(VEGETATION_TYPES)
)
# Type codes kept per vegetation variable, the most common ones in the data (the
# paper's 296 features hold 4 for high and 6 for low vegetation).
N_TYPE_CODES_KEPT = {"typeH": 4, "typeL": 6}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def decimal_year(times: pd.Series) -> np.ndarray:
    """Date as a fractional year, e.g. 2019-12-01 -> 2019.917."""
    times = pd.to_datetime(times)
    days = np.where(times.dt.is_leap_year, 366.0, 365.0)
    return (times.dt.year + (times.dt.dayofyear - 1) / days).to_numpy()


def environment(index: pd.DataFrame, names: list[str]) -> np.ndarray:
    """
    Environment variables E of each sample.

    Args:
        index: sample index with longitude, latitude and datetime
        names: from "longitude", "latitude" and "date" (fractional year)

    Returns:
        array of shape (len(index), len(names))
    """
    columns = {
        "longitude": index.longitude.to_numpy(float),
        "latitude": index.latitude.to_numpy(float),
        "date": decimal_year(index.datetime),
    }
    return np.stack([columns[n] for n in names], axis=1)


def icp_blocks(
    index: pd.DataFrame, table: pd.DataFrame, elevation_path: Path
) -> dict[str, np.ndarray]:
    """
    Feature block of every candidate variable.

    Args:
        index: sample index
        table: feature table from get_feature_table
        elevation_path: elevation grid for the altitude variable

    Returns:
        dict from each name in ICP_VARIABLES to an array of shape
        (len(index), n_features)
    """
    blocks = {}
    for variable in ICP_VARIABLES:
        if variable == ALTITUDE:
            blocks[variable] = altitude_features(index, elevation_path)
            continue
        x, _ = sample_features(index, table, [variable])
        if variable in VEGETATION_TYPES:
            common = np.argsort(-x.mean(axis=0))[: N_TYPE_CODES_KEPT[variable]]
            x = x[:, np.sort(common)]
        blocks[variable] = x
    return blocks


# ---------------------------------------------------------------------------
# Conditional independence test
# ---------------------------------------------------------------------------


def _delong_components(y: np.ndarray, p: np.ndarray):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tz = rankdata(np.concatenate([pos, neg]))
    tx, ty = rankdata(pos), rankdata(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return auc, v01, v10


def delong_test(y: np.ndarray, p_without: np.ndarray, p_with: np.ndarray) -> dict:
    """
    DeLong et al. (1988) test for the difference of two correlated AUCs.

    Args:
        y: binary labels
        p_without: scores of the model without the environment
        p_with: scores of the model with the environment

    Returns:
        dict with the z statistic, the one-tailed p-value of H0 "E does not
        improve the AUC" (pval_1tail), the two-tailed p-value (pval_2tail) and
        both AUCs (auc_noE, auc_E)
    """
    y = np.asarray(y)
    a0, v01_0, v10_0 = _delong_components(y, np.asarray(p_without, float))
    a1, v01_1, v10_1 = _delong_components(y, np.asarray(p_with, float))
    s01 = np.cov(np.stack([v01_0, v01_1]))
    s10 = np.cov(np.stack([v10_0, v10_1]))
    s = s01 / len(v01_0) + s10 / len(v10_0)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (a1 - a0) / np.sqrt(var) if var > 0 else 0.0
    return {
        "stat": float(z),
        "pval_1tail": float(norm.sf(z)),
        "pval_2tail": float(2 * norm.sf(abs(z))),
        "auc_E": float(a1),
        "auc_noE": float(a0),
    }


def oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    n_estimators: int,
    cfg: ICPConfig,
) -> np.ndarray:
    """
    Out-of-fold random forest probabilities.

    Args:
        x: features
        y: binary labels
        folds: fold of each sample
        n_estimators: number of trees
        cfg: forest settings (depth, class weight, seed)

    Returns:
        predicted probability of each sample from the forest not trained on it
    """
    prob = np.empty(len(y))
    for fold in np.unique(folds):
        test = folds == fold
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=cfg.max_depth,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=1,
        )
        rf.fit(x[~test], y[~test])
        prob[test] = rf.predict_proba(x[test])[:, 1]
    return prob


class ConditionalIndependenceTest:
    """
    Y _||_ E | X_S for subsets S of the candidate variables, with results cached
    in memory and, optionally, in a JSON-lines file so a run can be resumed.

    Args:
        blocks: feature block of each variable, see icp_blocks
        env: environment variables E
        y: binary labels
        folds: cross-validation fold of each sample
        cfg: ICP settings
        cache: JSON-lines file of previous results, appended to
    """

    def __init__(self, blocks, env, y, folds, cfg: ICPConfig, cache=None):
        self.blocks, self.env, self.y, self.folds = blocks, env, y, folds
        self.cfg = cfg
        self.order = {v: i for i, v in enumerate(blocks)}
        self.cache = None if cache is None else Path(cache)
        self.results = {}
        if self.cache is not None and self.cache.exists():
            for line in self.cache.read_text().splitlines():
                record = json.loads(line)
                self.results[self.key(record["subset"])] = record

    def key(self, subset) -> tuple[str, ...]:
        """Canonical (store-ordered) form of a subset."""
        return tuple(sorted(subset, key=self.order.__getitem__))

    def _compute(self, key) -> dict:
        x = np.concatenate([self.blocks[v] for v in key], axis=1)
        n = self.cfg.n_estimators
        p_without = oof_predictions(x, self.y, self.folds, n, self.cfg)
        x_env = np.concatenate([x, self.env], axis=1)
        p_with = oof_predictions(x_env, self.y, self.folds, n, self.cfg)
        return {"subset": list(key), **delong_test(self.y, p_without, p_with)}

    def __call__(self, subsets: list) -> list[dict]:
        """
        Test subsets, in parallel over cfg.n_jobs workers.

        Args:
            subsets: list of variable collections

        Returns:
            one result dict (see delong_test, plus "subset") per subset
        """
        keys = [self.key(s) for s in subsets]
        todo = list(dict.fromkeys(k for k in keys if k not in self.results))
        if todo:
            logging.info("Running %d conditional independence tests", len(todo))
            new = Parallel(n_jobs=self.cfg.n_jobs)(
                delayed(self._compute)(k) for k in todo
            )
            for record in new:
                self.results[tuple(record["subset"])] = record
            if self.cache is not None:
                with open(self.cache, "a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in new)
        return [self.results[k] for k in keys]


# ---------------------------------------------------------------------------
# ICP searches
# ---------------------------------------------------------------------------


def greedy_icp(test: ConditionalIndependenceTest, variables: list[str]) -> list[dict]:
    """
    Greedy ICP: repeatedly remove the variable whose removal gives the largest
    one-tailed p-value of Y _||_ E | X_S.

    Args:
        test: conditional independence test
        variables: initial set S

    Returns:
        one step per variable, in exclusion order, with the excluded variable,
        the p-value and AUCs of the test without it, and the variables left;
        the last variable has no test (p-value NaN)
    """
    remaining = list(variables)
    steps = []
    while len(remaining) > 1:
        candidates = [[v for v in remaining if v != u] for u in remaining]
        results = test(candidates)
        best = int(np.argmax([r["pval_1tail"] for r in results]))
        excluded = remaining.pop(best)
        steps.append({"excluded": excluded, "remaining": list(remaining)})
        steps[-1].update({k: v for k, v in results[best].items() if k != "subset"})
        logging.info(
            "Greedy ICP: exclude %s (p=%.4g)", excluded, results[best]["pval_1tail"]
        )
    nan = float("nan")
    steps.append(
        {
            "excluded": remaining[0],
            "remaining": [],
            **dict.fromkeys(
                ["stat", "pval_1tail", "pval_2tail", "auc_E", "auc_noE"], nan
            ),
        }
    )
    return steps


def greedy_causal_set(steps: list[dict], alpha: float) -> list[str]:
    """
    Variables not yet excluded when greedy ICP first rejects at level alpha.

    Args:
        steps: output of greedy_icp
        alpha: significance level

    Returns:
        the causal predictors, in exclusion order
    """
    for i, step in enumerate(steps):
        if step["pval_1tail"] < alpha:
            return [s["excluded"] for s in steps[i:]]
    return [steps[-1]["excluded"]]


def subsets_of_size(variables: list[str], min_size: int) -> list[list[str]]:
    """All subsets of variables with at least min_size elements."""
    return [
        list(c)
        for k in range(min_size, len(variables) + 1)
        for c in itertools.combinations(variables, k)
    ]


def intersection(sets: list[list[str]]) -> list[str]:
    """Variables in every set (empty when there is no set)."""
    if not sets:
        return []
    common = set(sets[0]).intersection(*sets[1:])
    return [v for v in sets[0] if v in common]


def defining_sets(accepted: list[list[str]], variables: list[str]) -> list[list[str]]:
    """
    Defining sets of Heinze-Deml et al. (2018): the minimal sets of variables
    that intersect every accepted set.

    Args:
        accepted: accepted subsets
        variables: candidate variables

    Returns:
        minimal hitting sets, by size then variable order
    """
    if not accepted:
        return []
    accepted = [set(a) for a in accepted]
    found = []
    for k in range(1, len(variables) + 1):
        for c in itertools.combinations(variables, k):
            s = set(c)
            if any(f <= s for f in found):
                continue
            if all(s & a for a in accepted):
                found.append(s)
    return [[v for v in variables if v in f] for f in found]


def exhaustive_icp(
    test: ConditionalIndependenceTest, subsets: list[list[str]], alpha: float
) -> dict:
    """
    Test every subset and intersect those accepted.

    Args:
        test: conditional independence test
        subsets: subsets to test
        alpha: significance level

    Returns:
        dict with the test results, the accepted subsets (p_1tail > alpha),
        their intersection and their defining sets
    """
    results = test(subsets)
    accepted = [r["subset"] for r in results if r["pval_1tail"] > alpha]
    variables = test.key({v for s in subsets for v in s})
    return {
        "n_tested": len(results),
        "n_accepted": len(accepted),
        "accepted": accepted,
        "intersection": intersection(accepted),
        "defining_sets": defining_sets(accepted, list(variables)),
        "tests": results,
    }


# ---------------------------------------------------------------------------
# HSIC clusters (Appendix A.2)
# ---------------------------------------------------------------------------


def _rbf_gram(x: np.ndarray) -> np.ndarray:
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    scale = max(np.median(d2), np.mean(d2), 1e-9)
    return np.exp(-d2 / scale)


def normalised_hsic(x: np.ndarray, z: np.ndarray) -> float:
    """
    Normalised HSIC (centred kernel alignment) of two multivariate samples with
    RBF kernels, as in pyrocast.icp.hsic.hsicRBF_jax.

    Args:
        x: array of shape (n, p)
        z: array of shape (n, q)

    Returns:
        value in [0, 1]; 0 for independent samples
    """
    n = len(x)
    h = np.eye(n) - 1.0 / n
    kx, kz = h @ _rbf_gram(x) @ h, h @ _rbf_gram(z) @ h
    return float((kx * kz).sum() / np.linalg.norm(kx) / np.linalg.norm(kz))


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1]; constant columns are left unchanged."""
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.where(hi > lo, (x - lo) / span, x)


def hsic_matrix(
    blocks: dict[str, np.ndarray], variables: list[str], n_samples: int, seed: int
) -> np.ndarray:
    """
    Normalised HSIC between the min-max scaled features of each pair of
    variables, on a random subset of samples.

    Args:
        blocks: feature block of each variable
        variables: variables to compare
        n_samples: samples used
        seed: seed for the sample choice

    Returns:
        symmetric matrix with NaN on the diagonal
    """
    first = blocks[variables[0]]
    rows = np.random.default_rng(seed).choice(len(first), n_samples, replace=False)
    scaled = [minmax(blocks[v])[rows] for v in variables]
    m = np.full((len(variables), len(variables)), np.nan)
    for i, j in itertools.combinations(range(len(variables)), 2):
        m[i, j] = m[j, i] = normalised_hsic(scaled[i], scaled[j])
    return m


def hsic_clusters(
    matrix: np.ndarray, variables: list[str], threshold: float
) -> list[list[str]]:
    """
    One cluster per variable: itself and every variable with HSIC >= threshold.

    Args:
        matrix: output of hsic_matrix
        variables: variables of the matrix rows
        threshold: dependence threshold

    Returns:
        clusters (not mutually exclusive), in variable order
    """
    return [
        [variables[i]]
        + [variables[j] for j in range(len(variables)) if matrix[i, j] >= threshold]
        for i in range(len(variables))
    ]


def cluster_subsets(
    clusters: list[list[str]], variables: list[str], min_size: int
) -> list[list[str]]:
    """
    Unique unions of every combination of at least min_size clusters.

    Args:
        clusters: output of hsic_clusters
        variables: variable order of the output subsets
        min_size: smallest number of clusters combined

    Returns:
        distinct subsets of variables
    """
    unions = {
        frozenset().union(*c)
        for k in range(min_size, len(clusters) + 1)
        for c in itertools.combinations(clusters, k)
    }
    subsets = [[v for v in variables if v in u] for u in unions]
    return sorted(subsets, key=lambda s: (len(s), [variables.index(v) for v in s]))


# ---------------------------------------------------------------------------
# Validation of the greedy ordering (Figure 3)
# ---------------------------------------------------------------------------


def _fold_aucs(x, y, folds, n_estimators, cfg) -> list[float]:
    prob = oof_predictions(x, y, folds, n_estimators, cfg)
    return [auc_or_nan(y[folds == k], prob[folds == k]) for k in np.unique(folds)]


def validate_ordering(
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    order: list[str],
    fold_schemes: dict[str, np.ndarray],
    cfg: ICPConfig,
) -> list[dict]:
    """
    Cross-validated AUC of forests on the variables left after each greedy step.

    Args:
        blocks: feature block of each variable
        y: binary labels
        order: greedy exclusion order
        fold_schemes: fold of each sample, per cross-validation scheme name
        cfg: ICP settings (validation_n_estimators trees)

    Returns:
        one record per (number of variables excluded, scheme) with the variables
        used, the per-fold AUCs, and their mean and population std
    """
    jobs = [
        (i, scheme, folds)
        for i in range(len(order))
        for scheme, folds in fold_schemes.items()
    ]
    aucs = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_fold_aucs)(
            np.concatenate([blocks[v] for v in order[i:]], axis=1),
            y,
            folds,
            cfg.validation_n_estimators,
            cfg,
        )
        for i, _, folds in jobs
    )
    return [
        {
            "n_excluded": i,
            "excluded": order[i - 1] if i else None,
            "scheme": scheme,
            "variables": order[i:],
            "fold_auc": a,
            "auc": float(np.nanmean(a)),
            "auc_std": float(np.nanstd(a)),
        }
        for (i, scheme, _), a in zip(jobs, aucs)
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _stage(path: Path, compute):
    """Load a stage result from path, or compute and save it."""
    if path.exists():
        logging.info("Loading %s", path)
        return json.loads(path.read_text())
    result = compute()
    path.write_text(json.dumps(result, indent=2))
    return result


def run(cfg: ICPExperimentConfig) -> dict:
    """
    Run greedy ICP, exhaustive ICP, ICP on HSIC clusters and the validation of
    the greedy ordering. Each stage is saved to the run directory and reloaded
    when the run is repeated; test results are cached in icp_tests.jsonl.

    Args:
        cfg: ICP experiment config

    Returns:
        summary metrics, also written to metrics.json
    """
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    m = cfg.model
    index = get_sample_index(cfg.data, cfg.mode)
    logging.info(
        "Sample index (%s): %d samples, %d positive, %d wildfires",
        cfg.mode,
        len(index),
        index.label.sum(),
        index.wildfire_id.nunique(),
    )
    table = get_feature_table(cfg.data, index)
    blocks = icp_blocks(index, table, cfg.data.elevation_path)
    y = index.label.to_numpy(dtype=int)
    folds = index.wildfire_id.map(wildfire_folds(index, cfg.split)).to_numpy()
    test = ConditionalIndependenceTest(
        blocks,
        environment(index, m.environment),
        y,
        folds,
        m,
        cache=run_dir / "icp_tests.jsonl",
    )
    variables = list(ICP_VARIABLES)

    full = test([variables])[0]
    steps = _stage(run_dir / "greedy.json", lambda: greedy_icp(test, variables))
    order = [s["excluded"] for s in steps]
    causal = greedy_causal_set(steps, m.alpha)

    selected = m.exhaustive_variables or order[-m.n_exhaustive :]
    selected = list(test.key(selected))
    exhaustive = _stage(
        run_dir / "exhaustive.json",
        lambda: {
            "variables": selected,
            **exhaustive_icp(
                test, subsets_of_size(selected, m.min_subset_size), m.alpha
            ),
        },
    )

    def clusters_stage():
        matrix = hsic_matrix(blocks, selected, m.hsic_samples, m.random_state)
        clusters = hsic_clusters(matrix, selected, m.hsic_threshold)
        subsets = cluster_subsets(clusters, selected, m.min_subset_size)
        return {
            "variables": selected,
            "hsic": matrix.tolist(),
            "clusters": clusters,
            **exhaustive_icp(test, subsets, m.alpha),
        }

    clustered = _stage(run_dir / "clusters.json", clusters_stage)

    def validation_stage():
        schemes = {}
        for scheme in ("event_cv", "spatial_cv"):
            split = SplitConfig(
                scheme=scheme, n_folds=cfg.split.n_folds, seed=cfg.split.seed
            )
            fold_of = wildfire_folds(index, split)
            schemes[scheme] = index.wildfire_id.map(fold_of).to_numpy()
        return validate_ordering(blocks, y, order, schemes, m)

    validation = _stage(run_dir / "validation.json", validation_stage)

    summary = {
        "n_samples": len(index),
        "n_positive": int(y.sum()),
        "n_events": int(index.event_id.nunique()),
        "n_wildfires": int(index.wildfire_id.nunique()),
        "n_features": int(sum(b.shape[1] for b in blocks.values())),
        "full_set_test": {k: v for k, v in full.items() if k != "subset"},
        "greedy_order": order,
        "greedy_pvalues": [s["pval_1tail"] for s in steps],
        "greedy_causal_set": causal,
        "exhaustive": {
            k: exhaustive[k]
            for k in ("variables", "n_tested", "n_accepted", "intersection")
        }
        | {"n_defining_sets": len(exhaustive["defining_sets"])},
        "clusters": {
            k: clustered[k]
            for k in ("clusters", "n_tested", "n_accepted", "intersection")
        },
        "validation": {
            scheme: {
                "auc": [v["auc"] for v in validation if v["scheme"] == scheme],
                "auc_std": [v["auc_std"] for v in validation if v["scheme"] == scheme],
            }
            for scheme in ("event_cv", "spatial_cv")
        },
    }
    save_run(run_dir, cfg, summary)
    logging.info("Greedy ICP causal set: %s", causal)
    return summary


def main(argv: list[str] | None = None) -> None:
    """
    Run every run of a YAML config (see utils.experiment.expand_grid), or one.

    Args:
        argv: command-line arguments, defaults to sys.argv
    """
    experiment_main(argv, ICPExperimentConfig, run, "Run invariant causal prediction")


if __name__ == "__main__":
    main()
