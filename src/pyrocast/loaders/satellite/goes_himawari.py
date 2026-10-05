"""
GOES satellite data loading and cropping.

Loads GOES-16/17 ABI L1b data from AWS S3 via Satpy,
crops to a region of interest, and returns xarray DataArrays.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_goes_da(
    dt, lat, lon, channel, lat_height=100, lon_width=100, goes_num=16, scan_mode=3
):
    """
    Load a cropped GOES ABI channel from AWS S3.

    Args:
        dt: datetime to retrieve
        lat: central latitude of region-of-interest
        lon: central longitude of region-of-interest
        channel: channel name, e.g. 'C01', 'C02', ..., 'C16'
        lat_height: half-height of crop in pixels
        lon_width: half-width of crop in pixels
        goes_num: GOES satellite number (16 or 17)
        scan_mode: ABI scan mode (3 or 6)

    Returns:
        tuple of (crop_scn, crop_lon, crop_lat) as xarray/numpy arrays
    """
    from satpy import Scene
    from satpy.readers import FSFile
    import fsspec

    path = f"noaa-goes{goes_num}/ABI-L1b-RadF/{dt.year:04}/{dt.strftime('%j')}/{dt.hour:02}"
    filename = (
        path
        + f"/OR_ABI-L1b-RadF-M{scan_mode:01}{channel}_G{goes_num:02}_s{dt.strftime('%Y%j%H')}{dt.minute:1}*"
    )

    the_files = fsspec.open_files("simplecache::s3://" + filename, s3={"anon": True})
    fs_files = [FSFile(open_file) for open_file in the_files]
    logger.info("Found %d files for %s", len(fs_files), filename)

    if not fs_files:
        raise ValueError(f"No files found: {filename}")

    scn = Scene(filenames=fs_files, reader="abi_l1b")
    scn.load([channel])
    chan_scn = scn.get(channel)

    lon_idx, lat_idx = chan_scn.attrs["area"].get_array_indices_from_lonlat(lon, lat)

    crop_scn = chan_scn[
        lat_idx - lat_height : lat_idx + lat_height,
        lon_idx - lon_width : lon_idx + lon_width,
    ]
    crop_lon, crop_lat = crop_scn.attrs["area"].get_lonlats_dask()

    return (
        crop_scn,
        crop_lon[
            lat_idx - lat_height : lat_idx + lat_height,
            lon_idx - lon_width : lon_idx + lon_width,
        ].compute(),
        crop_lat[
            lat_idx - lat_height : lat_idx + lat_height,
            lon_idx - lon_width : lon_idx + lon_width,
        ].compute(),
    )


def get_hima_da(
    dt, lat, lon, channel, px_dim=200, hima_num=8, cache_path="/tmp/himawari_cache"
):
    """
    Load a cropped Himawari AHI channel from AWS S3.

    Downloads segment files to a local cache, reads with Satpy,
    handles resolution differences between channels, and crops.

    Args:
        dt: datetime to retrieve
        lat: central latitude of region-of-interest
        lon: central longitude of region-of-interest
        channel: channel name, e.g. 'B01', 'B03', 'B07', 'B14', 'B16'
        px_dim: height and width of crop in pixels
        hima_num: Himawari satellite number (8 or 9)
        cache_path: local directory for temporary file downloads

    Returns:
        tuple of (crop_scn, crop_lon, crop_lat) as xarray/numpy arrays
    """
    import s3fs
    import glob
    import os
    import shutil
    from satpy import Scene

    channel_dict = {
        "B01": "10",
        "B02": "10",
        "B03": "05",
        "B04": "10",
        "B07": "20",
        "B14": "20",
        "B16": "20",
    }
    ch_res = channel_dict[channel]
    path = f"noaa-himawari{hima_num}/AHI-L1b-FLDK/{dt.year:04}/{dt.month:02}/{dt.day:02}/{dt.hour:02}{dt.minute:02}"
    filenames1 = path + "/*R" + ch_res + "*"
    fs = s3fs.S3FileSystem(anon=True)

    os.makedirs(cache_path, exist_ok=True)
    # clean cache before downloading
    for f in glob.glob(os.path.join(cache_path, "*")):
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

    logger.info("Downloading Himawari files from S3: %s", path)
    fs.get(filenames1, cache_path)
    filenames2 = glob.glob(os.path.join(cache_path, "*" + channel + "*"))
    if not filenames2:
        raise ValueError(f"No Himawari files found for {channel} at {path}")

    segments = [f.split("_S")[1][2:4] for f in filenames2]
    ch_seg = "10" if len(np.unique(segments)) > 1 else segments[0]
    filenames2 = glob.glob(
        os.path.join(cache_path, "*" + channel + "*S??" + ch_seg + "*")
    )

    if not filenames2:
        raise ValueError(f"No matching segment files for {channel} at {path}")

    scn = Scene(filenames=filenames2, reader="ahi_hsd")
    lat_height = px_dim // 2
    lon_width = px_dim // 2

    scn.load([channel])
    chan_scn = scn.get(channel)

    # clean cache after loading
    for f in glob.glob(os.path.join(cache_path, "*")):
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

    lon_idx, lat_idx = chan_scn.attrs["area"].get_array_indices_from_lonlat(lon, lat)

    if channel == "B03":
        chan_scn = chan_scn.coarsen(x=2, y=2, boundary="trim").mean()
        crop_scn = chan_scn[
            lat_idx // 2 - lat_height : lat_idx // 2 + lat_height,
            lon_idx // 2 - lon_width : lon_idx // 2 + lon_width,
        ]
    elif channel in ("B07", "B14", "B16"):
        crop_scn = chan_scn[
            lat_idx - lat_height // 2 : lat_idx + lat_height // 2 + 1,
            lon_idx - lon_width // 2 : lon_idx + lon_width // 2 + 1,
        ]
        x_int = np.arange(crop_scn.x.values.min(), crop_scn.x.values.max(), 1000)
        y_int = np.arange(crop_scn.y.values.min(), crop_scn.y.values.max(), 1000)
        crop_scn = crop_scn.interp(x=x_int, y=y_int, method="nearest")
        crop_scn = crop_scn.reindex(y=list(reversed(crop_scn.y)))
    else:
        crop_scn = chan_scn[
            lat_idx - lat_height : lat_idx + lat_height,
            lon_idx - lon_width : lon_idx + lon_width,
        ]

    crop_lon, crop_lat = crop_scn.attrs["area"].get_lonlats_dask()

    return (
        crop_scn,
        crop_lon[
            lat_idx - lat_height : lat_idx + lat_height,
            lon_idx - lon_width : lon_idx + lon_width,
        ].compute(),
        crop_lat[
            lat_idx - lat_height : lat_idx + lat_height,
            lon_idx - lon_width : lon_idx + lon_width,
        ].compute(),
    )
