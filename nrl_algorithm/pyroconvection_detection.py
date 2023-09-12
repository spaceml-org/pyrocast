import zarr
import numpy as np
from gcsfs import GCSFileSystem
import os
import copy
import numpy as np
import pandas as pd
from pysolar.solar import pys
from datetime import datetime, timedelta, timezone
import logging


def pyro_detection(dataset_cubes: np.array, datetime_list: list, lat: float, lon: float) -> tuple:
    """   
    Returns PyroCb flag binary flags and masks for N timestamps.

    Args:
        dataset_cubes (np.array):  N x 6 x 200 x 200
        datetime_list (list):
        lat (float): coordinate in °N
        lon (float): coordinate in °E

    Returns:
        tuple: _description_
    """

    # Set of flags that tests if there are any pixels in the data that pass each of the 5 tests below
    pyrocb_flag = np.zeros((len(dataset_cubes), 5)).astype(bool)

    # Mask of flags for each pixel that past the final 3 tests.
    # If all pixels are false, either the file wasn't found or it was taken at night/twilight
    pyrocb_flag_mask = []
    for i in range(len(dataset_cubes)):

        where_pyrocb_flag = np.zeros((3, 200, 200)).astype(bool)

        # Make sure file was read in
        pyrocb_flag[i, 0] = True

        # Daytime test: SZA < 80 degrees
        sza = datetime_to_sza(lat, lon, datetime_list[i])
        if sza >= 80.0:
            pyrocb_flag_mask.append(where_pyrocb_flag)
            continue
        else:
            pyrocb_flag[i, 1] = True

        # Deep convection test: Brightness temp at 11um < -20C
        pass_deep_convection_test = (dataset_cubes[i][4:5, :, :] < 273.15-20)
        if (~np.any(pass_deep_convection_test)):
            #logging.info(path + ' did not pass deep convection test')
            pyrocb_flag_mask.append(where_pyrocb_flag)
            continue
        else:
            where_pyrocb_flag[0, np.where(pass_deep_convection_test)[
                1], np.where(pass_deep_convection_test)[2]] = True
            pyrocb_flag[i, 2] = True

        # Cloud opacity test: Brightness temp at 11um - Brightness temp at 13 um < 3C
        tmp1 = copy.deepcopy(dataset_cubes[i][4:5, :, :])
        tmp2 = copy.deepcopy(dataset_cubes[i][5:6, :, :])
        # need to do things in a convoluted way in order to preserve dimensions
        tmp1[0, :, :] = tmp1[0, :, :]-tmp2[0, :, :]
        pass_cloud_opacity_test = (tmp1 < 3.0) &\
                                  (pass_deep_convection_test)
        if (~np.any(pass_cloud_opacity_test)):
            #logging.info(path + ' did not pass cloud opacity test')
            pyrocb_flag_mask.append(where_pyrocb_flag)
            continue
        else:
            where_pyrocb_flag[1, np.where(pass_cloud_opacity_test)[
                1], np.where(pass_cloud_opacity_test)[2]] = True
            pyrocb_flag[i, 3] = True

        # Skipping LCL test - doesn't seem to matter that much anyway?

        # Standard cloud microphysics test: Brightness temp at 4um - Brightness temp at 11 um > 50C
        tmp1 = copy.deepcopy(dataset_cubes[i][3:4, :, :])
        tmp2 = copy.deepcopy(dataset_cubes[i][4:5, :, :])
        # need to do things in a convoluted way in order to preserve dimensions
        tmp1[0, :, :] = tmp1[0, :, :]-tmp2[0, :, :]
        pass_cloud_microphysics_test = copy.deepcopy(pass_cloud_opacity_test)
        pass_cloud_microphysics_test[0, :, :] = (tmp1[0, :, :] > 50.0) &\
            (pass_cloud_opacity_test[0, :, :])
        if (~np.any(pass_cloud_microphysics_test)):
            logging.info(path + ' did not pass cloud microphysics test')
            pyrocb_flag_mask.append(where_pyrocb_flag)
            continue
        else:
            where_pyrocb_flag[2, np.where(pass_cloud_microphysics_test)[
                1], np.where(pass_cloud_microphysics_test)[2]] = True
            logging.info(path + ' - we have a PyroCb!')
            pyrocb_flag_mask.append(where_pyrocb_flag)
            pyrocb_flag[i, 4] = True

    return pyrocb_flag, pyrocb_flag_mask


def output_zarr(pyrocb_flag, pyrocb_flag_mask, datetime_list, event_id):

    save_dir = str(event_id) + '/'

    images = []
    for idataset, masky in enumerate(pyrocb_flag_mask):
        if (len(masky) > 0):

            year = str(datetime_list[idataset].year)
            month = datetime_list[idataset].strftime("%m")
            day = str(datetime_list[idataset].day)
            time = datetime_list[idataset].strftime(
                "%H") + datetime_list[idataset].strftime("%M") + '00'
            time_str = year + month + day + time

            mask_path = 'data/' + save_dir + time_str + 'PyroCb_mask.zarr'
            flag_path = 'data/' + save_dir + time_str + 'PyroCb_flags.zarr'

            zarr.save(mask_path, masky)
            zarr.save(flag_path, pyrocb_flag[idataset, :])

            # move to GCP
            gcp_path = 'gs://eu-aerosols-landing/PyroCb_masks/'
            move_command1 = 'gsutil cp -r ' + flag_path + ' ' + gcp_path + save_dir
            move_command2 = 'gsutil cp -r ' + mask_path + ' ' + gcp_path + save_dir
            os.system(move_command1)
            os.system(move_command2)

    # delete files in folder
    # remove last slash in path string
    del_command = 'rm -rf ' + 'data/' + save_dir[:-1]
    os.system(del_command)


def create_datetime_list(date, frequency=10.0):
    """ List of datetimes for a given day and frequency. """

    # Convert to datetime object
    ns = 1e-9  # number of seconds in a nanosecond
    start_time = datetime.utcfromtimestamp(date.astype(int) * ns)

    end_time = start_time + timedelta(hours=24)
    datetime_list = [start_time]
    delta = timedelta(minutes=frequency)
    new_t = start_time

    while new_t < end_time:
        new_t = new_t + delta
        datetime_list.append(new_t)

    time_vector = np.arange(start_time.hour + start_time.minute/60.0,
                            end_time.hour + end_time.minute/60.0 + frequency/60.0, frequency/60.0)

    return datetime_list, time_vector


def datetime_to_sza(lat, lon, dt):
    dobj = dt.replace(tzinfo=timezone.utc)
    sza = float(90) - pys.get_altitude(lat, lon, dobj)
    return sza


# Example

if __name__ == "__main__":

    fs = GCSFileSystem()

    # import zarr file

    bucket = 'eu-aerosols-landing'
    outdir = 'WildFireSubevents'
    storage_root = 'gs://{}/{}/'.format(bucket, outdir)
    event_id = "180_1"

    data_path = os.path.join(storage_root, event_id, "data")
    print(data_path)
    za = zarr.load(fs.get_mapper(data_path))

    event_df = pd.read_csv('pyrocb_labels_all_2022_07_14.csv')
    event_df['date'] = pd.to_datetime(event_df['date'])

    # filter by ID
    event = event_df[event_df['id'] == 180]
    pyro_id = event_df['id'].values[0]
    lat = event['latitude'].values[0]
    lon = event['longitude'].values[0]
    date = event['date'].values[0]
    datetime_list, _ = create_datetime_list(date, frequency=60.0)

    # GOES17 channels: [1,2,3,7,14,16]
    channel_idx = np.array([0, 1, 2, 6, 13, 15])

    i = 0
    np.all(za[i, channel_idx, :, :] != 0)
    hour_idx, = np.where([np.all(za[i, channel_idx, :, :] != 0)
                          for i in range(za.shape[0])])

    # Create data cube
    data_list = []
    for h in hour_idx:

        cube_list = []
        for ch in channel_idx:
            ch_arr = za[h, ch, :, :]
            cube_list.append(ch_arr)
        cube_arr = np.array(cube_list)
        data_list.append(cube_arr)

    data_arr = np.array(data_list)

    # Apply NRL algorithm
    pyrocb_flags, pyrocb_flag_masks = pyro_detection(
        data_arr, datetime_list, lat, lon)

    output_zarr(pyrocb_flags, pyrocb_flag_masks, datetime_list, event_id)
