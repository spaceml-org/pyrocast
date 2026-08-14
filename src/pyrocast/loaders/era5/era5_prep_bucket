# Once the raw ERA5 data is downloaded (era5_loader.py):
# 1. Prepare the bucket with the zarr file by running this script
# 2. Run the loop to crop, reproject and save the data in 'era5_processing.py'

from gcsfs import GCSFileSystem
import contextlib
import joblib
from joblib import delayed
from joblib import Parallel
import fsspec
import zarr
import numpy as np

import pickle5
import os
from tqdm import tqdm

import fsspec
import pandas as pd
from utils.sql_utils import *


def prepare_zarr(bucket, outdir, df, num_days, num_bands, freq_smpl):
    # df : the dataframe with the wildfires

    wildfirepieces = df['wildfire_piece_id']
    event_ids = wildfirepieces.values.tolist()
    OUTPUT_DIR = 'gs://{}/{}/'.format(bucket, outdir)

    gcp_prepare_archive(event_ids,
                        OUTPUT_DIR,
                        n_jobs=-1,
                        verbose=0,
                        patch_size=200,
                        timesteps=(60/freq_smpl)*24*num_days,
                        bands=num_bands)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def create_zarr_structure(fs_mapper, storage_root, event_id, timesteps=12*24*2, patch_size=200, bands=6):
    """

    :param str storage_path: The path to the zarr array
    :param str event_id: The name of the wildfire id
    :param int timesteps: The number of timesteps each event will contain
    :param int patch_size: The image size (assumed x=y)
    :param int bands: The number of bands to store
    :return:
    """
    import xarray as xr
    import dask.array

    data_path = os.path.join(storage_root, event_id, "data")

    zarr.open_array(
        fs_mapper(data_path),
        "w",
        shape=(timesteps, bands, patch_size, patch_size),
        chunks=(1, 1, patch_size, patch_size),
        dtype=np.float32,
    )


def gcp_prepare_archive(
    event_ids,
    storage_root: str,
    n_jobs: int = -1,
    verbose: int = 0,
    patch_size=200,
    timesteps=12*24*2,
    bands=20,
    **kwargs,
) -> bool:

    fs = GCSFileSystem()

    # FIXME - I don't know what this loop is doing, it looks like it's just calling zarr.open on all the files
    #   but I don't know why it would need to do that - the next code block will deal with this?
    # This creates the zarr directories (I think in SatExtractor/preparer they also had some logic to check whether it should overwrite or not?)

    print(len(event_ids))
    with tqdm_joblib(
        tqdm(
            desc=f"parallel building zarr tile roots on {storage_root}",
            total=len(event_ids))):
        Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(
            [
                delayed(zarr.open)(fs.get_mapper(
                    os.path.join(storage_root, event_id)), "a")
                for event_id in event_ids
            ],
        )

    #logger.info(f"parallel building zarr archives on {storage_root}")
    jobs = []
    for event_id in event_ids:
        jobs.append(
            delayed(create_zarr_structure)(
                fs.get_mapper,
                storage_root,
                event_id,
                timesteps=timesteps,
                patch_size=patch_size,
                bands=bands
            ),
        )

    with tqdm_joblib(
        tqdm(desc="Building Archives.", total=len(event_ids)),
    ):
        Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(jobs)

    return True


if __name__ in "__main__":

    fs = fsspec.filesystem("gs")
    conn = fs.open(
        "eu-aerosols-dev-features/wildfires_to_extract/wildfires.pkl")
    #wildfires = pd.read_csv(wildfires)
    wildfires = pickle5.load(conn)  # open( wildfires, "rb" )

    # transform into a wildfire piece dataframew with one chunk of time (1 day), location
    num_days = 1  # size of pieces
    wildfire_pieces = blowUpWildfires(wildfires, num_days)
    print("number of wildfire pieces: ", wildfire_pieces.shape)

    num_bands = 19
    freq_smpl = 60  # units in minutes

    bucket = 'eu-aerosols-landing'
    outdir = 'ERA5/crop_data2'

    prepare_zarr(bucket, outdir, wildfire_pieces,
                 num_days, num_bands, freq_smpl)
