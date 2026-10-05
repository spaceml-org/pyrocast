import os

import numpy as np
import pandas as pd
import zarr

from pyrocast.config import Mode

N_HOURS = 24
LEAD_HOURS = 6
N_NRL_TESTS = 5
# NRL states of the Pyrocast paper (Figure 1), keyed by nrl_state: the index of
# the last NRL test passed, 0 when none is. States 0 and 1 are both "none".
NRL_STATE_NAMES = {
    0: "none",
    1: "none",
    2: "convection",
    3: "deep_convection",
    4: "pyrocb",
}


def load_flag(flag_path):
    """
    Load a PyroCb flag array, returning None if it does not exist.

    zarr.load creates an empty group for a missing path with zarr>=3, so the
    path is checked explicitly rather than relying on zarr.load returning None.
    """
    if not os.path.isdir(flag_path):
        return None
    return zarr.load(flag_path)


def event_flag_arrays(event_id: str, flag_root: str) -> np.ndarray:
    """
    Read the NRL test results for every hour of an event.

    Flag files are numbered from 01, so file HH holds the flags of cube hour HH - 1.

    Args:
        event_id: wildfire piece id, e.g. "100_1"
        flag_root: directory holding one folder of flag files per event

    Returns:
        int array of shape (N_HOURS, N_NRL_TESTS): 1 where an NRL test passed,
        0 where it failed, -1 for hours whose flag file is missing. The last
        test is the PyroCb flag.
    """
    flags = np.full((N_HOURS, N_NRL_TESTS), -1, dtype=int)
    for hour in range(N_HOURS):
        path = os.path.join(flag_root, event_id, f"{hour + 1:02d}_PyroCb_flags.zarr")
        flag = load_flag(path)
        if flag is not None:
            flags[hour] = np.asarray(flag, dtype=int)[:N_NRL_TESTS]
    return flags


def event_flags(event_id: str, flag_root: str) -> np.ndarray:
    """
    Read the PyroCb flag for every hour of an event.

    Args:
        event_id: wildfire piece id, e.g. "100_1"
        flag_root: directory holding one folder of flag files per event

    Returns:
        int array of shape (N_HOURS,): 1 or 0, or -1 where the flag is missing
    """
    return event_flag_arrays(event_id, flag_root)[:, -1]


def nrl_states(flag_arrays: np.ndarray) -> np.ndarray:
    """
    Reduce NRL test results to one state per hour, as get_full_flag in the NRL
    algorithm: the index of the last test passed, or 0 when none is.

    Args:
        flag_arrays: array of shape (n, N_NRL_TESTS) from event_flag_arrays

    Returns:
        int array of shape (n,) in 0..N_NRL_TESTS - 1, -1 where flags are missing
    """
    passed = flag_arrays == 1
    last = N_NRL_TESTS - 1 - np.argmax(passed[:, ::-1], axis=1)
    states = np.where(passed.any(axis=1), last, 0)
    return np.where(flag_arrays[:, 0] < 0, -1, states)


def match_samples(
    hourly: pd.DataFrame, mode: Mode, lead_hours: int = LEAD_HOURS
) -> pd.DataFrame:
    """
    Pair input hours with target flags for a prediction mode.

    - detection: inputs at t, label is the flag at t
    - forecast: inputs at t, label is the flag at t + lead_hours
    - forecast_oracle: geostationary at t and ERA5 at t + lead_hours (a perfect
      weather forecast), label is the flag at t + lead_hours

    Args:
        hourly: one row per (event_id, date_idx) with a flag column, -1 if missing,
            and optionally an nrl_state column
        mode: prediction mode
        lead_hours: forecast lead time in hours

    Returns:
        the input rows with a missing target dropped, the flag and nrl_state
        columns renamed to flag_now and state_now (at the input hour, -1 if
        missing), and added columns era5_idx (ERA5 hour to read) and label
        (target flag)
    """
    shift = 0 if mode == "detection" else lead_hours
    flags = hourly.set_index(["event_id", "date_idx"]).flag
    target = pd.MultiIndex.from_arrays([hourly.event_id, hourly.date_idx + shift])
    renamed = {"flag": "flag_now", "nrl_state": "state_now"}
    samples = hourly.rename(columns=renamed).assign(
        era5_idx=hourly.date_idx + (shift if mode == "forecast_oracle" else 0),
        label=flags.reindex(target, fill_value=-1).to_numpy(),
    )
    return samples[samples.label >= 0].reset_index(drop=True)
