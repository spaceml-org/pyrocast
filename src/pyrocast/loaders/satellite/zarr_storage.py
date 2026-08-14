"""
Zarr storage utilities for satellite data patches.

Stores cropped satellite patches into pre-allocated zarr arrays on GCS.
"""

import os
import logging
import numpy as np
import zarr
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def store_patch(fs_mapper, patch, patchlon, patchlat, storage_path, event_id, band_idx, dt_idx):
    """
    Store a satellite patch into a pre-allocated zarr array.

    Args:
        fs_mapper: filesystem mapper (e.g. gcsfs.GCSFileSystem().get_mapper)
        patch: xarray DataArray with the satellite data
        patchlon: longitude array for the patch
        patchlat: latitude array for the patch
        storage_path: root storage path
        event_id: wildfire event identifier
        band_idx: spectral band index
        dt_idx: datetime index
    """
    data_path = os.path.join(storage_path, event_id, "data")
    arr = zarr.open_array(store=fs_mapper(data_path), dtype=np.float32)

    arr[dt_idx, band_idx, :, :] = patch.to_numpy()

    # store lon/lat on band 0 to avoid concurrent writes
    if band_idx == 0:
        arr[dt_idx, 18, :, :] = patchlon
        arr[dt_idx, 19, :, :] = patchlat


def format_sql_date(dt_str):
    """Parse a SQL-formatted datetime string into a timezone-aware datetime.

    Args:
        dt_str: string like '2019-01-15 12:30:00'

    Returns:
        datetime with UTC timezone
    """
    parts = dt_str.split(" ")
    date_parts = parts[0].split("-")
    time_parts = parts[1].split(":")
    return datetime(
        int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),
        int(time_parts[0]), int(time_parts[1]), int(time_parts[2]),
        tzinfo=timezone.utc,
    )
