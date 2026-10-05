"""
Spatial summary features of the geostationary and ERA5 cubes.

Both Pyrocast papers summarise each 200x200 field by 11 statistics: the mean,
standard deviation, minimum, maximum and the 1, 5, 25, 50, 75, 95 and 99th
percentiles. The features of every (event, hour) are computed once and cached in
DataConfig.feature_cache, so that every mode and input set reuses them.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

from pyrocast.config import ERA5_VARIABLES, DataConfig, Mode
from pyrocast.utils.data.dataload import get_sample_index, read_cubes

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
STATS = ("mean", "std", "min", "max") + tuple(f"p{q}" for q in PERCENTILES)
N_STATS = len(STATS)
GEO_VARIABLES = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")
# Wind speeds added in the ICP paper, from the u and v components.
WIND_SPEEDS = {"uv10": ("u10", "v10"), "uv250": ("u", "v")}
CUBE_VARIABLES = GEO_VARIABLES + ERA5_VARIABLES + tuple(WIND_SPEEDS)
ALTITUDE = "alt"
# Categorical ERA5 vegetation types, summarised by the fraction of pixels of each
# type code instead of the 11 statistics. Interpolation to the geostationary grid
# blurs the codes, so values are rounded to the nearest integer.
VEGETATION_TYPES = {"typeH": "tvh", "typeL": "tvl"}
N_TYPE_CODES = 21
PIXEL_KM = 1.0
KM_PER_DEGREE = 111.32
INDEX_KEYS = ["event_id", "date_idx"]


def stat_columns(variable: str) -> list[str]:
    """Feature table columns of a summarised variable, in STATS order."""
    return [f"{variable}__{stat}" for stat in STATS]


def type_columns(variable: str) -> list[str]:
    """Feature table columns of a vegetation type variable, one per type code."""
    return [f"{variable}__c{code}" for code in range(N_TYPE_CODES)]


TABLE_COLUMNS = [c for v in CUBE_VARIABLES for c in stat_columns(v)] + [
    c for v in VEGETATION_TYPES for c in type_columns(v)
]


def summary_features(cubes: np.ndarray) -> np.ndarray:
    """
    Summarise each channel of each cube by the 11 statistics in STATS.

    Args:
        cubes: array of shape (n, channels, height, width)

    Returns:
        array of shape (n, channels * N_STATS), grouped by channel
    """
    flat = cubes.reshape(cubes.shape[0], cubes.shape[1], -1)
    stats = [flat.mean(axis=-1), flat.std(axis=-1), flat.min(-1), flat.max(-1)]
    stats += list(np.percentile(flat, PERCENTILES, axis=-1))
    return np.stack(stats, axis=-1).reshape(cubes.shape[0], -1)


def type_fractions(fields: np.ndarray) -> np.ndarray:
    """
    Fraction of pixels of each vegetation type code.

    Args:
        fields: array of shape (n, height, width) of type codes

    Returns:
        array of shape (n, N_TYPE_CODES)
    """
    codes = np.clip(np.rint(fields), 0, N_TYPE_CODES - 1).astype(int)
    codes = codes.reshape(len(codes), -1)
    counts = np.stack([np.bincount(c, minlength=N_TYPE_CODES) for c in codes])
    return counts / codes.shape[1]


def _event_table(
    cfg: DataConfig, event_id: str, satellite: str, hours: list[int]
) -> pd.DataFrame:
    cubes = read_cubes(cfg, event_id, hours, satellite, "both")
    geo, era5 = cubes[:, : len(GEO_VARIABLES)], cubes[:, len(GEO_VARIABLES) :]
    speeds = [
        np.hypot(
            era5[:, ERA5_VARIABLES.index(u)],
            era5[:, ERA5_VARIABLES.index(v)],
        )
        for u, v in WIND_SPEEDS.values()
    ]
    stats = summary_features(np.concatenate([geo, era5, np.stack(speeds, 1)], 1))
    types = [
        type_fractions(era5[:, ERA5_VARIABLES.index(v)])
        for v in VEGETATION_TYPES.values()
    ]
    values = np.concatenate([stats, *types], axis=1).astype(np.float32)
    table = pd.DataFrame(values, columns=TABLE_COLUMNS)
    table.insert(0, "date_idx", hours)
    table.insert(0, "event_id", event_id)
    return table


def compute_feature_table(cfg: DataConfig, rows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the summary features of (event, hour) pairs from the cubes.

    Args:
        cfg: data configuration
        rows: event_id, date_idx and satellite of each pair

    Returns:
        DataFrame with INDEX_KEYS and TABLE_COLUMNS, one row per pair
    """
    rows = rows.drop_duplicates(INDEX_KEYS)
    groups = list(rows.groupby("event_id", sort=True))
    tables = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_event_table)(
            cfg, event_id, g.satellite.iloc[0], sorted(g.date_idx.tolist())
        )
        for event_id, g in groups
    )
    if not tables:
        return pd.DataFrame(columns=INDEX_KEYS + TABLE_COLUMNS)
    return pd.concat(tables, ignore_index=True)


def _needed_rows(index: pd.DataFrame) -> pd.DataFrame:
    geo = index[["event_id", "date_idx", "satellite"]]
    era5 = index[["event_id", "era5_idx", "satellite"]].rename(
        columns={"era5_idx": "date_idx"}
    )
    return pd.concat([geo, era5]).drop_duplicates(INDEX_KEYS)


def _read_feature_cache(cache: Path) -> pd.DataFrame | None:
    if not cache.exists():
        return None
    table = pd.read_pickle(cache)
    if list(table.columns) != INDEX_KEYS + TABLE_COLUMNS:
        logging.warning("Rebuilding feature cache with an outdated layout: %s", cache)
        return None
    return table


def get_feature_table(cfg: DataConfig, index: pd.DataFrame) -> pd.DataFrame:
    """
    Load the summary features of every hour that the samples read.

    Features missing from cfg.feature_cache are computed and appended to it.

    Args:
        cfg: data configuration
        index: sample index (event_id, date_idx, era5_idx, satellite)

    Returns:
        DataFrame indexed by (event_id, date_idx) with TABLE_COLUMNS, covering
        the geostationary hours (date_idx) and ERA5 hours (era5_idx) of index
    """
    needed = _needed_rows(index)
    cache = None if cfg.feature_cache is None else Path(cfg.feature_cache)
    table = None if cache is None else _read_feature_cache(cache)
    if table is None:
        table = pd.DataFrame(columns=INDEX_KEYS + TABLE_COLUMNS)
    have = pd.MultiIndex.from_frame(table[INDEX_KEYS])
    missing = needed[~pd.MultiIndex.from_frame(needed[INDEX_KEYS]).isin(have)]
    if len(missing):
        logging.info("Computing features of %d event hours", len(missing))
        new = compute_feature_table(cfg, missing)
        table = new if table.empty else pd.concat([table, new], ignore_index=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            table.to_pickle(tmp)
            os.replace(tmp, cache)
    return table.set_index(INDEX_KEYS)


def load_elevation(path: Path) -> RegularGridInterpolator:
    """
    Load a global elevation grid as an interpolator of (lat, lon in [0, 360)).

    Args:
        path: NetCDF file with a data variable on lat and lon coordinates

    Returns:
        bilinear interpolator returning metres
    """
    ds = xr.open_dataset(path, decode_times=False)
    da = ds["data"].squeeze(drop=True).sortby("lat").sortby("lon")
    lon = da.lon.values % 360
    order = np.argsort(lon)
    lon, values = lon[order], da.values[:, order]
    # wrap around the dateline
    lon = np.concatenate([lon[-1:] - 360, lon, lon[:1] + 360])
    values = np.concatenate([values[:, -1:], values, values[:, :1]], axis=1)
    return RegularGridInterpolator((da.lat.values, lon), values, bounds_error=False)


def altitude_field(
    elevation: RegularGridInterpolator, longitude: float, latitude: float, size: int
) -> np.ndarray:
    """
    Elevation of a size x size grid of PIXEL_KM pixels centred on a location.

    Args:
        elevation: interpolator from load_elevation
        longitude: centre longitude in degrees
        latitude: centre latitude in degrees
        size: pixels per side

    Returns:
        array of shape (size, size) in metres
    """
    offsets = (np.arange(size) - (size - 1) / 2) * PIXEL_KM / KM_PER_DEGREE
    lats = latitude + offsets
    lons = longitude + offsets / np.cos(np.deg2rad(latitude))
    grid_lat, grid_lon = np.meshgrid(lats, lons % 360, indexing="ij")
    return elevation((grid_lat, grid_lon))


def altitude_features(
    index: pd.DataFrame, elevation_path: Path, size: int = 200
) -> np.ndarray:
    """
    Summary features of the elevation around each sample's wildfire.

    Args:
        index: sample index with event_id, longitude and latitude
        elevation_path: elevation grid, see load_elevation
        size: pixels per side of the area summarised

    Returns:
        array of shape (len(index), N_STATS)
    """
    elevation = load_elevation(elevation_path)
    events = index.drop_duplicates("event_id").set_index("event_id")
    fields = np.stack(
        [
            altitude_field(elevation, e.longitude, e.latitude, size)
            for e in events.itertuples()
        ]
    )
    stats = pd.DataFrame(summary_features(fields[:, None]), index=events.index)
    return stats.loc[index.event_id].to_numpy(dtype=np.float32)


def sample_features(
    index: pd.DataFrame, table: pd.DataFrame, variables: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    Look up the summary features of each sample.

    Geostationary channels are taken at the sample's date_idx and ERA5 variables
    (including wind speeds and vegetation types) at its era5_idx.

    Args:
        index: sample index
        table: feature table from get_feature_table
        variables: names from CUBE_VARIABLES or VEGETATION_TYPES, in output order

    Returns:
        feature array of shape (len(index), n_features) and its column names
    """
    geo_rows = pd.MultiIndex.from_arrays([index.event_id, index.date_idx])
    era5_rows = pd.MultiIndex.from_arrays([index.event_id, index.era5_idx])
    blocks, names = [], []
    for variable in variables:
        if variable in VEGETATION_TYPES:
            columns = type_columns(variable)
        elif variable in CUBE_VARIABLES:
            columns = stat_columns(variable)
        else:
            raise KeyError(f"Unknown feature variable: {variable}")
        rows = geo_rows if variable in GEO_VARIABLES else era5_rows
        blocks.append(table.loc[rows, columns].to_numpy(dtype=np.float32))
        names += columns
    return np.concatenate(blocks, axis=1), names


def input_variables(inputs: str, era5_variables: list[str] | None) -> list[str]:
    """
    Feature variables of an experiment's inputs.

    Args:
        inputs: "geostationary", "era5" or "both"
        era5_variables: ERA5 subset, None for all 19

    Returns:
        variable names in store order: geostationary channels, then ERA5
    """
    variables = []
    if inputs in ("geostationary", "both"):
        variables += list(GEO_VARIABLES)
    if inputs in ("era5", "both"):
        chosen = set(era5_variables or ERA5_VARIABLES)
        variables += [v for v in ERA5_VARIABLES if v in chosen]
    return variables


def main(argv: list[str] | None = None) -> None:
    """
    Fill the feature cache for every mode of an experiment config's data, so
    that experiments sharing the cache can then run in parallel.

    Args:
        argv: command-line arguments, defaults to sys.argv
    """
    parser = argparse.ArgumentParser(description="Build the Pyrocast feature cache")
    parser.add_argument("--config", required=True, help="experiment YAML config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    with open(args.config) as f:
        cfg = DataConfig.model_validate(yaml.safe_load(f)["data"])
    if cfg.feature_cache is None:
        raise ValueError("data.feature_cache is not set")
    index = pd.concat([get_sample_index(cfg, mode) for mode in get_args(Mode)])
    table = get_feature_table(cfg, index)
    logging.info("Feature cache %s holds %d event hours", cfg.feature_cache, len(table))


if __name__ == "__main__":
    main()
