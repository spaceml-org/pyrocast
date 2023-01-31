from google.cloud import bigquery
from tqdm import tqdm
from gcsfs import GCSFileSystem
import fsspec
from datetime import datetime, timedelta
import pickle5
import zarr
from datetime import timedelta, timezone, datetime
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch import nn
from google.cloud import storage
import os
import copy
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np


def load_data(train_list: list, parent_dir: str, event_id: str):
    """
    Loads and matches datacubes and flags.

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
    path_list, dataset_cubes, flags = load_files2(
        parent_dir, datetime_list, event_id, frequency=10)

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


class HimawariDataset(torch.utils.data.Dataset):
    """ Dataset object withs data, labels and transformations """

    def __init__(self, dataset_cubes, flags, transform=None, target_transform=None):
        self.img_labels = flags
        self.img_cubes = dataset_cubes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = np.moveaxis(self.img_cubes[idx], 0, -1)
        label = int(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def load_loaders(train_list: list, test_list: list, parent_dir: str, event_id: str) -> torch.utils.data.DataLoader:
    """  """

    _, train_dataset_cubes, train_flags = load_data(
        train_list, parent_dir, event_id)
    _, test_dataset_cubes, test_flags = load_data(
        test_list, parent_dir, event_id)

    D = train_dataset_cubes.shape[1]

    # Transform values generated from means and variances of training set
    mu = [np.mean(train_dataset_cubes[:, i, :, :])
          for i in range(D)]
    sigma = [np.std(train_dataset_cubes[:, i, :, :])
             for i in range(D)]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mu, sigma)])

    batch_size = 64

    trainset = HimawariDataset(dataset_cubes=np.array(
        train_dataset_cubes), flags=train_flags, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = HimawariDataset(
        dataset_cubes=test_dataset_cubes, flags=test_flags, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def load_rfdata(train_list, test_list, parent_dir: str, event_id: str):
    """ Load data and calculates statistics for RF model """

    xtrain, ytrain, xtest, ytest = [1, 1, 1, 1, ]

    return xtrain, ytrain, xtest, ytest


def match_image_label(event_id, date_str, satellite, flag_root):
    """
    For a given event, returns information about matched data.

    Inputs:
        event_id (str), sub event id for pyrocb wildfire
        date_str (str), in the SQL table format
        satellite (str), satelitte name
        flag_root (str), where to load flag info from 

    Outputs:
    (5 list of equal size)

        final_datacube_dateindxs (list of int), indices to keep for corresponding geostationnary and ERA5 zarr datacubes
        flag_list (list of booleans), flag 6 hours ahead
        flag_now_list (list of booleans), current flag
        final_datetime_list (list of datetimes)
        satellite_list (list of str), 
    """

    # generate datetime list inside
    start_datetime = datetime.strptime(date_str[:13], '%Y-%m-%d %H')
    datetime_list, _ = create_daytime_list([start_datetime], frequency=60)

    # Match flags 6 hours ahead
    flag_list = []
    flag_now_list = []

    # setup a array of indices and removes those with no corresponding flag information
    datacube_dateindxs = np.arange(0, 24).tolist()

    # make list of indices to remove (we know that some indices will definitely be out of range)
    pop_list = np.arange(18, 24).tolist()

    # iterate over hours in datacube
    hour_num = 6

    for t in range(18):
        hour_num = hour_num + 1
        hour_str = str(hour_num).zfill(2)
        hour_now_str = str(hour_num-6).zfill(2)

        flag_path = os.path.join(
            flag_root, event_id, hour_str + '_PyroCb_flags.zarr')
        flag_path_now = os.path.join(
            flag_root, event_id, hour_now_str + '_PyroCb_flags.zarr')

        flag_za = zarr.load(flag_path)
        flag_za_now = zarr.load(flag_path_now)
        if flag_za_now is None:
            flag_za_now = -1
        else:
            flag_za_now = flag_za_now[4]

        if flag_za is None:
            pop_list.append(t)
        if flag_za is not None:
            # print(flag_za)
            flag_list.append(flag_za[4])
            flag_now_list.append(flag_za_now)
            # print(flag_path)

    final_datacube_dateindxs = np.delete(datacube_dateindxs, pop_list)
    final_datetime_list = np.delete(datetime_list, pop_list)
    satellite_list = [satellite for _ in range(len(flag_list))]

    return final_datacube_dateindxs, flag_list, flag_now_list, final_datetime_list, satellite_list


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
