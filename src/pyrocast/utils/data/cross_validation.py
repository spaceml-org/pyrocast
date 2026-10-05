"""
Train/test splits grouped by wildfire.

The Pyrocast papers evaluate with 5-fold cross-validation where every
observation of a wildfire is in the same fold: "event" CV assigns wildfires to
folds at random and "spatial" CV clusters them by k-means on their latitude and
longitude, to estimate performance in an unseen region.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit

from pyrocast.config import SplitConfig

GROUP_COLUMN = "wildfire_id"


def wildfire_folds(index: pd.DataFrame, split: SplitConfig) -> pd.Series:
    """
    Assign each wildfire to a cross-validation fold.

    Args:
        index: sample index with wildfire_id, longitude and latitude columns
        split: split configuration with scheme "event_cv" or "spatial_cv"

    Returns:
        fold number (0 to n_folds - 1) indexed by sorted wildfire_id
    """
    fires = index.groupby(GROUP_COLUMN)[["longitude", "latitude"]].mean()
    if len(fires) < split.n_folds:
        raise ValueError(
            f"{len(fires)} wildfires cannot be split into {split.n_folds} folds"
        )
    if split.scheme == "event_cv":
        order = np.random.default_rng(split.seed).permutation(len(fires))
        folds = np.empty(len(fires), dtype=int)
        folds[order] = np.arange(len(fires)) % split.n_folds
    elif split.scheme == "spatial_cv":
        coords = fires[["latitude", "longitude"]].to_numpy()
        coords = (coords - coords.mean(0)) / coords.std(0)
        kmeans = KMeans(n_clusters=split.n_folds, n_init=10, random_state=split.seed)
        folds = kmeans.fit_predict(coords)
    else:
        raise ValueError(f"Not a cross-validation scheme: {split.scheme}")
    return pd.Series(folds, index=fires.index, name="fold")


def fold_splits(
    index: pd.DataFrame, split: SplitConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Train and test row positions of each fold, keeping each wildfire in one set.

    Args:
        index: sample index with wildfire_id, longitude and latitude columns
        split: split configuration

    Returns:
        one (train, test) pair of row position arrays per fold; a single pair
        for the "holdout" scheme
    """
    if split.scheme == "holdout":
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=split.test_fraction, random_state=split.seed
        )
        return [next(splitter.split(index, groups=index[GROUP_COLUMN]))]
    folds = index[GROUP_COLUMN].map(wildfire_folds(index, split)).to_numpy()
    rows = np.arange(len(index))
    return [(rows[folds != k], rows[folds == k]) for k in range(split.n_folds)]


def split_by_fire(
    index: pd.DataFrame, split: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the sample index once into train and test sets, keeping each wildfire
    in one set.

    Args:
        index: sample index with a wildfire_id column
        split: split configuration; test_fraction and seed are used

    Returns:
        train and test sample indices
    """
    holdout = split.model_copy(update={"scheme": "holdout"})
    train, test = fold_splits(index, holdout)[0]
    return (
        index.iloc[train].reset_index(drop=True),
        index.iloc[test].reset_index(drop=True),
    )
