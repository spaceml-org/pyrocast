"""Shared fixtures: a small synthetic Pyrocast data root on disk."""

import os

import numpy as np
import pandas as pd
import pytest
import zarr

SIZE = 200
N_HOURS_WRITTEN = 12
VALID_EVENTS = {
    "1_1": ("GOES16", "2020-01-01 18:00:00+00:00"),
    "1_2": ("GOES16", "2020-01-02 18:00:00+00:00"),
    "2_1": ("Himawari", "2020-02-01 21:00:00+00:00"),
    "2_2": ("Himawari", "2020-02-02 21:00:00+00:00"),
    "3_1": ("GOES17", "2020-03-01 18:00:00+00:00"),
    "3_2": ("GOES17", "2020-03-02 18:00:00+00:00"),
}
SIX_CHANNEL_EVENT = "4_1"
NO_ERA5_EVENT = "5_1"
# pyrocb_events.csv rows: fire id -> (wildfire id or None, country). Fire 2 has no
# GlobFire match, so it forms its own wildfire "pyrocb_2".
FIRES = {
    1: ("W1", "Australia"),
    2: (None, "US"),
    3: ("W3", "Canada"),
    4: ("W4", "Russia"),
    5: ("W5", "Russia"),
}


def nrl_flag(hour):
    """NRL test results of flag file hour: state 4 (pyroCb) on even hours, else
    state 2 when hour % 3 == 0, else state 1."""
    flag = np.zeros(5, dtype=bool)
    flag[:2] = True
    flag[2] = hour % 3 == 0
    flag[4] = hour % 2 == 0
    return flag


def _write_cube(path, n_channels, offset):
    arr = zarr.create_array(
        store=path,
        shape=(24, n_channels, SIZE, SIZE),
        chunks=(1, 1, SIZE, SIZE),
        dtype="f4",
        fill_value=0.0,
        zarr_format=2,
    )
    values = np.arange(N_HOURS_WRITTEN)[:, None] * 100 + np.arange(n_channels)
    arr[:N_HOURS_WRITTEN] = np.broadcast_to(
        (values + offset)[:, :, None, None].astype("f4"),
        (N_HOURS_WRITTEN, n_channels, SIZE, SIZE),
    )


def _write_flags(flag_root, event_id):
    for hour in range(1, 13):
        flag = nrl_flag(hour)
        path = os.path.join(flag_root, event_id, f"{hour:02d}_PyroCb_flags.zarr")
        zarr.save_array(path, flag, zarr_format=2)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """Data root with six valid events from three fires plus two invalid events.

    Cube values are hour * 100 + channel + i, where i is the event's position in
    VALID_EVENTS (then the invalid events), and event i is at longitude 100 + i,
    latitude -30 + i.
    """
    root = tmp_path_factory.mktemp("pyrocast_data")
    geo = root / "Geostationary_imagery"
    era5 = root / "climate_and_fuel"
    flags = root / "PyroCb_flags_and_masks"
    rows = []
    all_events = {
        **VALID_EVENTS,
        SIX_CHANNEL_EVENT: ("GOES16", "2020-04-01 18:00:00+00:00"),
        NO_ERA5_EVENT: ("GOES16", "2020-05-01 18:00:00+00:00"),
    }
    for i, (event_id, (satellite, start)) in enumerate(all_events.items()):
        n_geo = 6 if event_id == SIX_CHANNEL_EVENT else 20
        _write_cube(str(geo / event_id / "data"), n_geo, offset=i)
        if event_id != NO_ERA5_EVENT:
            _write_cube(str(era5 / event_id / "data"), 19, offset=i)
        _write_flags(str(flags), event_id)
        fire, piece = event_id.split("_")
        for date_idx, snapshot in [(2, "2099-01-01 00:00:00+00:00"), (1, start)]:
            rows.append(
                {
                    "pyrocb_id": int(fire),
                    "piece_id": int(piece),
                    "wildfire_piece_id": event_id,
                    "longitude": 100.0 + i,
                    "latitude": -30.0 + i,
                    "snapshot": snapshot,
                    "date_idx": date_idx,
                    "satellite": satellite,
                }
            )
    pd.DataFrame(rows).to_csv(root / "wildfire_snapshots.csv", index=False)
    pd.DataFrame(
        {
            "pyroCb_id": list(FIRES),
            "wildfire_id": [w for w, _ in FIRES.values()],
            "country": [c for _, c in FIRES.values()],
            "region": "somewhere",
        }
    ).to_csv(root / "pyrocb_events.csv", index=False)
    return root


def listing(root):
    """Sorted relative paths of all files under root."""
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(root)
        for f in fs
    )
