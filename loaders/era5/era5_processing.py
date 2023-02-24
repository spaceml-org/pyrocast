# Once the raw ERA5 data is downloaded:
# 1. Prepare the bucket with the zarr file by running 'era5_prep_bucket.py'
# 2. Run the loop to crop, reproject and save the data


import os
import zarr
import fsspec
import pickle5

import xarray as xr
import numpy as np

from tqdm import tqdm
from gcsfs import GCSFileSystem
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime, timedelta
from utils.sql_utils import *


def load_era5_files(time: datetime, lon: float, variable: str) -> xr.DataArray:
    """ Load files of interests from wildfire datetimes, returns filename list.

    Args:
        time (datetime): wildfire event datetime
        lon (float): longitude of event
        variable (str): variable to load

    Returns:
        xr.DataArray: _description_
    """

    year = str(time.year)

    if lon[0, 0] > 0:
        a = 'australia'
    if lon[0, 0] < 0:
        a = 'north_america'

    filename = variable + '_' + year + '.nc'

    if os.path.exists('data/' + a + '/' + filename) == False:
        raw_dir = 'gs://eu-aerosols-landing/ERA5/raw_data/'
        raw_filepath = raw_dir + a + '/' + filename
        print('downloading: ', raw_filepath)
        dl_command = 'gsutil -m cp ' + raw_filepath + ' data/' + a + '/' + filename
        os.system(dl_command)

    t_end = time + timedelta(hours=23)
    da1 = xr.open_dataset('data/' + a + '/' + filename)
    da2 = da1.sel(time=slice(time, t_end))
    return da2


def save_zarr(event_id, arr, storage_path):
    data_path = os.path.join(storage_path, event_id, "data")
    empty_arr = zarr.open_array(store=fs_mapper(data_path), dtype=np.float32,)
    empty_arr[:, :, :, :] = arr


def create_query():
    """ Load fires sql table """
    query = """
    SELECT
        pyrocb_id,
        piece_id,
        wildfire_piece_id,
        longitude,
        latitude,
        snapshot,
        date_idx,
        band, 
        satellite,
        scanmode,
        status,
        job_group
    FROM
        `eu-aerosols.pyrocb.fires`
             """

    clause = "WHERE piece_id > 0"

    sort_clause = """
        ORDER BY 
            pyroCb_id, 
            piece_id, 
            date_idx,
            band,
            status
                  """

    return query + clause + sort_clause


# LOOP

# Define directories to load and save data
fs = GCSFileSystem()
fs_mapper = fs.get_mapper
bucket = 'eu-aerosols-landing'
indir = 'WildFireSubevents'  # 'WildFireDataFlow'
download_root = 'gs://{}/{}/'.format(bucket, indir)
output_dir = 'ERA5/crop_data'
storage_path = 'gs://{}/{}/'.format(bucket, output_dir)

# Create event list
query = create_query()
bqclient = bigquery.Client()

event_df = (
    bqclient.query(query)
    .result()
    .to_dataframe(
        # Optionally, explicitly request to use the BigQuery Storage API. As of
        # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
        # API is used by default.
        create_bqstorage_client=True,
    )
)

# Only keep wildfire events that were succesfully dowloaded
event_df2 = event_df[event_df['status'] ==
                     'success'].drop_duplicates('wildfire_piece_id')

# Make event lists
event_list = event_df2['wildfire_piece_id'].values.tolist()
date_list = event_df2['snapshot'].tolist()

# Variable list
variable_list = ['10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_gust_since_previous_post_processing',
                 'boundary_layer_height', 'convective_available_potential_energy', 'convective_inhibition',
                 'geopotential', 'surface_latent_heat_flux', 'surface_sensible_heat_flux',
                 'relative_humidity', 'vertical_velocity', 'u_component_of_wind',
                 'v_component_of_wind', 'fuel']

# Loop over events
fail_list = []

for i in tqdm(range(len(event_list))):

    event_id = event_list[i]
    date_str = str(date_list[i])

    # check if cropped ERA5 file already exists
    name = os.path.join(output_dir, event_id, 'data/0.0.0.0')
    storage_client = storage.Client()
    bucket_dir = storage_client.bucket(bucket)
    stats = storage.Blob(bucket=bucket_dir, name=name).exists(storage_client)

    if stats == False:

        # load zarray file
        data_path = os.path.join(download_root, event_id, "data")
        za = zarr.load(fs.get_mapper(data_path))

        try:
            # check which hours indices have lat, lon info
            for i in range(24):
                if np.any(za[i, 19] != 0):
                    lat = za[i, 19]
                if np.any(za[i, 18] != 0):
                    lon = za[i, 18]

            if lat is not None:
                if lon is not None:

                    # create empty template
                    empty_arr = np.ones_like(lon)
                    proj_da = xr.DataArray(empty_arr, coords=dict(longitude=(
                        ('x', 'y'), lon), latitude=(('x', 'y'), lat)), dims=('x', 'y'))

                    var_ds_list = []

                    for v in variable_list:

                        date = datetime.strptime(date_str[:13], '%Y-%m-%d %H')

                        # load era5 data and merge Datasets
                        era5_ds = load_era5_files(date, v, lon, lat)

                        # interpolate
                        reproj_ds = era5_ds.interp(
                            latitude=proj_da.latitude, longitude=proj_da.longitude,  method='linear')
                        var_ds_list.append(reproj_ds)

                    # merge images and split the relative humidity values over pressure levels
                    big_ds = xr.merge(var_ds_list)
                    big_ds = big_ds.assign(r650=big_ds.r.sel(level=650),
                                           r750=big_ds.r.sel(level=750),
                                           r850=big_ds.r.sel(level=850))
                    big_ds = big_ds.drop('r')
                    big_ds = big_ds.drop('level')

                    # to zarr
                    arr = big_ds.to_array().values
                    arr = arr.transpose((1, 0, 2, 3))

                    # save zarr on GCP
                    save_zarr(event_id, arr, storage_path)
        except:
            print('error')
