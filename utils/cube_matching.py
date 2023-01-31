from datetime import datetime
import zarr
import os
import numpy as np
import utils.dataload as dl


def forecast_match(event_id, date_str, satellite, flag_root):
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
    datetime_list, _ = dl.create_daytime_list([start_datetime], frequency=60)

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


def oracle_match(event_id, date_str, satellite, flag_root):

    return 0


def detection_match(event_id, date_str, satellite, flag_root):

    return 0
