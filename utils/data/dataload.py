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
    _, dataset_cubes, flags = load_files(
        parent_dir, datetime_list, event_id)

    # Get rid of missing and nighttime/twilight data
    dataset_cubes_train = []
    flags_train = []
    datetimes_train = []
    dataset_cubes_night = []
    flags_night = []
    datetimes_night = []

    for i in range(len(dataset_cubes)):

        if (flags[i][0]):
            # Set aside nighttime and sunrise/sunset data
            if (flags[i][1]):
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
    """ Create datetime list from event dates """

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

        time_vector = np.arange(start_time.hour + start_time.minute/60.0,
                                end_time.hour + end_time.minute/60.0 + frequency/60.0, frequency/60.0)
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

    logging.info('Start reading in files')
    # if you are working from Google Cloud Console
    #client = storage.Client()

    # filename components
    secs = '00'
    instrument = '_himawari_'

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
        flagtime = flag_datetime.strftime(
            "%H") + flag_datetime.strftime("%M") + secs

        # Store the different timesteps
        save_dir = parent_dir + year + month + day + time + '/'
        path_list.append(save_dir)

        # For each timestep, read in all the spectral channels
        channel_datasets = []
        prefix = 'Himawari8/CroppedImg/' + year + month + day + time
        #file_list = client.list_blobs('eu-aerosols-landing', prefix=prefix)
        file_list = glob.glob('eu-aerosols-landing', prefix=prefix)

        for blob in file_list:
            filename = str(blob.name)
            try:
                tmp = pd.read_csv('eu-aerosols-landing/' + filename, header=None).values
                # tmp = pd.read_csv('gs://eu-aerosols-landing/' +filename, header=None).values
                channel_datasets.append(tmp)
            except FileNotFoundError:
                logging.warning('File not found: eu-aerosols-landing/' + filename)
                # logging.warning('File not found: gs://eu-aerosols-landing/' + filename)

        # For each timestep, if the complete set of channels was found, concatenate them together in an xarray
        if len(channel_datasets) != 6:
            logging.warning('Channels not found: ' + prefix)
            path_list.pop()

        if len(channel_datasets) == 6:
            output_as_dataarray = xr.concat([
                xr.DataArray(
                    X,
                    dims=["record", "edge"],
                    coords={"record": range(X.shape[0]), "edge": range(X.shape[1])},)
                for X in channel_datasets],
                dim="descriptor",).assign_coords(descriptor=["B01", "B03", "B04", "B07", "B14", "B16"])
            dataset_cubes.append(output_as_dataarray.values)

            # Read in corresponding PyroCb flag. Only one label per observation
            # flagpath = 'gs://eu-aerosols-landing/PyroCb_masks/' + event_id + \
            flagpath = 'eu-aerosols-landing/PyroCb_masks/' + event_id + \
                '/' + year + month + str(int(day)) + \
                flagtime + 'PyroCb_flags.zarr'

            try:
                mask_flag = zarr.load(flagpath)
                # Only last flag relevant for each observation (the others pertain to the earlier tests in the NRL algorithm)
                flags.append(mask_flag)

            except:
                print('Flag not found: ' + flagpath)
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

    if satellite == 'Himawari':
        datacube = geo_za[np.array(date_idx)[:, None],
                          np.array([0, 2, 3, 6, 13, 15])]
    if satellite == 'GOES16':
        datacube = geo_za[np.array(date_idx)[:, None],
                          np.array([0, 1, 2, 6, 13, 15])]
    if satellite == 'GOES17':
        datacube = geo_za[np.array(date_idx)[:, None],
                          np.array([0, 1, 2, 6, 13, 15])]

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
    datacube = era5_za[date_idx, ]

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
