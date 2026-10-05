from tqdm import tqdm
from datetime import timedelta
import zarr
import os
import copy
import xarray as xr
import pandas as pd
import numpy as np
import logging
import glob
from pathlib import Path

from joblib import Parallel, delayed

from pyrocast.config import DataConfig, Inputs, Mode
import pyrocast.utils.data.cube_matching as cm

# uncomment if you are working from Google Cloud Console
# from google.cloud import storage


def load_data(train_list: list, parent_dir: str, event_id: str):
    """
    Discards nighttime data.

    Args:
        train_list (list): _description_
        parent_dir (str): _description_
        event_id (str): _description_

    Returns:
        datetimes_list
        dataset_cubes
        flags
    """

    # Convert start, end and timestep to a list of datetimes and a vector of decimal values
    datetime_list, _ = create_daytime_list(train_list)

    # Load files
    _, dataset_cubes, flags = load_files(parent_dir, datetime_list, event_id)

    # Get rid of missing and nighttime/twilight data
    dataset_cubes_train = []
    flags_train = []
    datetimes_train = []
    dataset_cubes_night = []
    flags_night = []
    datetimes_night = []

    for i in range(len(dataset_cubes)):

        if flags[i][0]:
            # Set aside nighttime and sunrise/sunset data
            if flags[i][1]:
                dataset_cubes_train.append(dataset_cubes[i])
                flags_train.append(flags[i][-1])
                datetimes_train.append(datetime_list[i])
            else:
                dataset_cubes_night.append(dataset_cubes[i])
                flags_night.append(flags[i][-1])
                datetimes_night.append(datetime_list[i])

    dataset_cubes = copy.deepcopy(dataset_cubes_train)
    datetimes_list = copy.deepcopy(datetimes_train)
    flags = copy.deepcopy(flags_train)

    return datetimes_list, dataset_cubes, flags


def create_daytime_list(datetimes, frequency=10.0):
    """Create datetime list from event dates"""

    datetime_list = []
    time_vector_list = []
    delta = timedelta(minutes=frequency)

    for dt in datetimes:
        start_time = dt
        end_time = dt + timedelta(hours=24)

        new_t = start_time
        while new_t < end_time:
            datetime_list.append(new_t)
            new_t = new_t + delta

        time_vector = np.arange(
            start_time.hour + start_time.minute / 60.0,
            end_time.hour + end_time.minute / 60.0 + frequency / 60.0,
            frequency / 60.0,
        )
        time_vector_list.append(time_vector)

    return datetime_list, time_vector_list


def load_files(parent_dir, datetime_list, event_id):
    """
    Load files of interests from wildfire datetimes, returns filename list.
    Args:
        parent_dir (str)
        datetime_list (list)
        frequency (int)
    """

    logging.info("Start reading in files")
    # if you are working from Google Cloud Console
    # client = storage.Client()

    # filename components
    secs = "00"
    instrument = "_himawari_"

    path_list = []
    dataset_cubes = []
    flags = []

    for t in tqdm(datetime_list):

        year = str(t.year)
        month = t.strftime("%m")
        day = t.strftime("%d")
        time = t.strftime("%H") + t.strftime("%M") + secs

        # + 6 hours for flags
        flag_datetime = t + timedelta(hours=6)
        flagtime = flag_datetime.strftime("%H") + flag_datetime.strftime("%M") + secs

        # Store the different timesteps
        save_dir = parent_dir + year + month + day + time + "/"
        path_list.append(save_dir)

        # For each timestep, read in all the spectral channels
        channel_datasets = []
        prefix = "Himawari8/CroppedImg/" + year + month + day + time
        # file_list = client.list_blobs('eu-aerosols-landing', prefix=prefix)
        file_list = glob.glob("eu-aerosols-landing", prefix=prefix)

        for blob in file_list:
            filename = str(blob.name)
            try:
                tmp = pd.read_csv("eu-aerosols-landing/" + filename, header=None).values
                # tmp = pd.read_csv('gs://eu-aerosols-landing/' +filename, header=None).values
                channel_datasets.append(tmp)
            except FileNotFoundError:
                logging.warning("File not found: eu-aerosols-landing/" + filename)
                # logging.warning('File not found: gs://eu-aerosols-landing/' + filename)

        # For each timestep, if the complete set of channels was found, concatenate them together in an xarray
        if len(channel_datasets) != 6:
            logging.warning("Channels not found: " + prefix)
            path_list.pop()

        if len(channel_datasets) == 6:
            output_as_dataarray = xr.concat(
                [
                    xr.DataArray(
                        X,
                        dims=["record", "edge"],
                        coords={"record": range(X.shape[0]), "edge": range(X.shape[1])},
                    )
                    for X in channel_datasets
                ],
                dim="descriptor",
            ).assign_coords(descriptor=["B01", "B03", "B04", "B07", "B14", "B16"])
            dataset_cubes.append(output_as_dataarray.values)

            # Read in corresponding PyroCb flag. Only one label per observation
            # flagpath = 'gs://eu-aerosols-landing/PyroCb_masks/' + event_id + \
            flagpath = (
                "eu-aerosols-landing/PyroCb_masks/"
                + event_id
                + "/"
                + year
                + month
                + str(int(day))
                + flagtime
                + "PyroCb_flags.zarr"
            )

            try:
                mask_flag = zarr.load(flagpath)
                # Only last flag relevant for each observation (the others pertain to the earlier tests in the NRL algorithm)
                flags.append(mask_flag)

            except:
                print("Flag not found: " + flagpath)
                path_list.pop()

    return path_list, dataset_cubes, flags


def getImage(geostationary_root, event_id, date_idx, satellite):
    # print(geostationary_root)
    geo_path = os.path.join(geostationary_root, event_id, "data")
    geo_za = zarr.load(geo_path)
    # Extract channels to datacubes
    # if date_idx==0:
    # print(geo_za.shape)

    # print(geo_za.shape)

    if satellite == "Himawari":
        datacube = geo_za[np.array(date_idx)[:, None], np.array([0, 2, 3, 6, 13, 15])]
    if satellite == "GOES16":
        datacube = geo_za[np.array(date_idx)[:, None], np.array([0, 1, 2, 6, 13, 15])]
    if satellite == "GOES17":
        datacube = geo_za[np.array(date_idx)[:, None], np.array([0, 1, 2, 6, 13, 15])]

    return datacube


def getImageCube(data_key, data_files, i, geostationary_root):
    """
    Make dataset cubes from:

    data_key (str)
    data_files (
    i (int), index for
    geostationary_root(str)
    """
    j = data_key["level1_key"][i]
    k = data_key["level2_key"][i]

    event_id = data_files["event_id"][j]
    date_idx = data_files["date_idx"][j][k]
    satellite = data_files["satellite"][j][k]

    datacube = getImage(geostationary_root, event_id, [date_idx], satellite)

    datacube = np.nan_to_num(datacube)

    return datacube


def getERA5(era5_root, event_id, date_idx):
    # print(era5_root)
    era5_path = os.path.join(era5_root, event_id, "data")
    era5_za = zarr.load(era5_path)

    # Extract channels to datacubes
    datacube = era5_za[date_idx,]

    return datacube


def getERA5Cube(data_key, data_files, i, era5_root):
    """
    Make ERA5 cubes from:

    data_key (str)
    data_files (
    i (int), index for
    geostationary_root(str)
    """
    j = data_key["level1_key"][i]
    k = data_key["level2_key"][i]

    event_id = data_files["event_id"][j]
    date_idx = data_files["date_idx"][j][k]  # +6

    datacube = getERA5(era5_root, event_id, date_idx)

    datacube = np.nan_to_num(datacube)

    return datacube


GEO_CHANNELS = {
    "Himawari": [0, 2, 3, 6, 13, 15],
    "GOES16": [0, 1, 2, 6, 13, 15],
    "GOES17": [0, 1, 2, 6, 13, 15],
}
N_ERA5_CHANNELS = 19
HOURLY_COLUMNS = [
    "event_id",
    "fire_id",
    "date_idx",
    "datetime",
    "satellite",
    "longitude",
    "latitude",
    "flag",
    "nrl_state",
]


def read_event_table(snapshots_path: Path) -> pd.DataFrame:
    """
    Read one row per wildfire piece from the snapshots CSV.

    Args:
        snapshots_path: path to wildfire_snapshots.csv

    Returns:
        DataFrame indexed by event_id with fire_id, start, satellite, longitude
        and latitude, where start is the snapshot of the first date index.
    """
    df = pd.read_csv(
        snapshots_path,
        usecols=[
            "pyrocb_id",
            "wildfire_piece_id",
            "longitude",
            "latitude",
            "snapshot",
            "date_idx",
            "satellite",
        ],
    )
    first = df.sort_values("date_idx").groupby("wildfire_piece_id").first()
    first = first.rename(columns={"pyrocb_id": "fire_id", "snapshot": "start"})
    first.index.name = "event_id"
    return first[["fire_id", "start", "satellite", "longitude", "latitude"]]


def read_wildfires(events_path: Path) -> pd.DataFrame:
    """
    Read the wildfire and country of each pyroCb event.

    PyroCb events that were not matched to a GlobFire wildfire form a wildfire
    of their own, as in the Pyrocast papers.

    Args:
        events_path: path to pyrocb_events.csv

    Returns:
        DataFrame indexed by fire_id (the pyroCb id) with columns wildfire_id
        (str) and country
    """
    df = pd.read_csv(
        events_path,
        usecols=["pyroCb_id", "wildfire_id", "country"],
        dtype={"wildfire_id": str},
    )
    df["wildfire_id"] = df.wildfire_id.fillna("pyrocb_" + df.pyroCb_id.astype(str))
    return df.rename(columns={"pyroCb_id": "fire_id"}).set_index("fire_id")


def _open_array(path: Path) -> zarr.Array:
    return zarr.open_array(str(path), mode="r")


def _has_valid_cubes(cfg: DataConfig, event_id: str) -> bool:
    try:
        geo = _open_array(cfg.geostationary_path / event_id / "data")
        era5 = _open_array(cfg.era5_path / event_id / "data")
    except (FileNotFoundError, ValueError, KeyError, zarr.errors.BaseZarrError):
        return False
    return geo.shape[1] > max(max(c) for c in GEO_CHANNELS.values()) and (
        era5.shape[1] == N_ERA5_CHANNELS
    )


def available_events(cfg: DataConfig) -> list[str]:
    """
    List events present in the snapshots table and all three stores.

    Events whose cubes cannot be opened or have too few channels are skipped.

    Args:
        cfg: data configuration

    Returns:
        sorted event ids from cfg.countries if set, truncated to cfg.max_events
        if set
    """
    stores = [cfg.geostationary_path, cfg.era5_path, cfg.flags_path]
    table = read_event_table(cfg.snapshots_path)
    if cfg.countries is not None:
        country = table.fire_id.map(read_wildfires(cfg.events_path).country)
        table = table[country.isin(cfg.countries)]
    candidates = set(table.index)
    for store in stores:
        candidates &= {p.name for p in store.iterdir() if p.is_dir()}
    events = []
    for event_id in sorted(candidates):
        if _has_valid_cubes(cfg, event_id):
            events.append(event_id)
        else:
            logging.warning("Skipping event with invalid cubes: %s", event_id)
        if cfg.max_events is not None and len(events) == cfg.max_events:
            break
    return events


def _event_hours(event_id: str, event: pd.Series, flags_root: str) -> pd.DataFrame:
    start = pd.to_datetime(event.start[:13], format="%Y-%m-%d %H")
    flags = cm.event_flag_arrays(event_id, flags_root)
    return pd.DataFrame(
        {
            "event_id": event_id,
            "fire_id": event.fire_id,
            "date_idx": np.arange(cm.N_HOURS),
            "datetime": start + pd.to_timedelta(np.arange(cm.N_HOURS), unit="h"),
            "satellite": event.satellite,
            "longitude": event.longitude,
            "latitude": event.latitude,
            "flag": flags[:, -1],
            "nrl_state": cm.nrl_states(flags),
        },
        columns=HOURLY_COLUMNS,
    )


def build_hourly_flags(cfg: DataConfig) -> pd.DataFrame:
    """
    Build the table of PyroCb flags: one row per (event, hour) of every valid event.

    Args:
        cfg: data configuration

    Returns:
        DataFrame with columns HOURLY_COLUMNS. flag and nrl_state are -1 when
        the flag file is missing.
    """
    events = read_event_table(cfg.snapshots_path)
    event_ids = available_events(cfg)
    frames = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_event_hours)(e, events.loc[e], str(cfg.flags_path)) for e in event_ids
    )
    if not frames:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_sample_index(cfg: DataConfig, mode: Mode) -> pd.DataFrame:
    """
    Build the table of samples for a prediction mode without using the cache.

    Args:
        cfg: data configuration
        mode: prediction mode, see cube_matching.match_samples

    Returns:
        sample index as returned by cube_matching.match_samples, with the
        wildfire_id and country of each event
    """
    return _samples(build_hourly_flags(cfg), cfg, mode)


def _samples(hourly: pd.DataFrame, cfg: DataConfig, mode: Mode) -> pd.DataFrame:
    """Select the configured events from the hourly flags and match samples."""
    wildfires = read_wildfires(cfg.events_path)
    unknown = set(hourly.fire_id) - set(wildfires.index)
    if unknown:
        raise ValueError(f"Fires missing from {cfg.events_path}: {sorted(unknown)}")
    hourly = hourly.join(wildfires, on="fire_id")
    if cfg.countries is not None:
        hourly = hourly[hourly.country.isin(cfg.countries)]
    if cfg.max_events is not None:
        keep = sorted(hourly.event_id.unique())[: cfg.max_events]
        hourly = hourly[hourly.event_id.isin(keep)]
    return cm.match_samples(hourly.reset_index(drop=True), mode)


def _read_hourly_cache(cache: Path) -> pd.DataFrame | None:
    if not cache.exists():
        return None
    hourly = pd.read_csv(cache, dtype={"event_id": str})
    if list(hourly.columns) != HOURLY_COLUMNS:
        logging.warning("Rebuilding index cache with an outdated layout: %s", cache)
        return None
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    return hourly


def get_sample_index(cfg: DataConfig, mode: Mode) -> pd.DataFrame:
    """
    Load the samples for a prediction mode, using the hourly flags cached in
    cfg.index_cache and building the cache if it is missing or outdated.

    The cache holds the flags of all events for all modes, so cfg.countries and
    cfg.max_events are applied after building or loading it and a cache is never
    silently truncated.

    Args:
        cfg: data configuration
        mode: prediction mode, see cube_matching.match_samples

    Returns:
        sample index as returned by cube_matching.match_samples, with the
        wildfire_id and country of each event
    """
    if cfg.index_cache is None:
        return build_sample_index(cfg, mode)

    cache = Path(cfg.index_cache)
    hourly = _read_hourly_cache(cache)
    if hourly is None:
        everything = {"max_events": None, "countries": None}
        hourly = build_hourly_flags(cfg.model_copy(update=everything))
        cache.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(cache, index=False)
    return _samples(hourly, cfg, mode)


def read_cubes(
    cfg: DataConfig,
    event_id: str,
    date_idxs: list[int],
    satellite: str,
    inputs: Inputs,
    era5_idxs: list[int] | None = None,
    era5_channels: list[int] | None = None,
) -> np.ndarray:
    """
    Read the hours of one event from the geostationary and/or ERA5 stores.

    Only the requested hours and channels are read. NaNs are set to zero.

    Args:
        cfg: data configuration
        event_id: wildfire piece id, e.g. "100_1"
        date_idxs: hour indices into the event cubes
        satellite: satellite name, selects the geostationary channels
        inputs: which stores to read; "both" stacks geostationary then ERA5
        era5_idxs: hour indices into the ERA5 cube, if different from date_idxs
        era5_channels: ERA5 channels to read, in this order; None reads all

    Returns:
        float32 array of shape (len(date_idxs), channels, height, width)
    """
    idxs = np.asarray(date_idxs, dtype=int)
    cubes = []
    if inputs in ("geostationary", "both"):
        geo = _open_array(cfg.geostationary_path / event_id / "data")
        cubes.append(geo.oindex[idxs, GEO_CHANNELS[satellite]])
    if inputs in ("era5", "both"):
        era5 = _open_array(cfg.era5_path / event_id / "data")
        if era5_idxs is not None:
            idxs = np.asarray(era5_idxs, dtype=int)
        channels = slice(None) if era5_channels is None else list(era5_channels)
        cubes.append(era5.oindex[idxs, channels])
    return np.nan_to_num(np.concatenate(cubes, axis=1).astype(np.float32))
