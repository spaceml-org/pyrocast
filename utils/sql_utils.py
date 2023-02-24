# utilities for creating big query database from pandas dataframe

import pytz
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
#from sqlalchemy import create_engine
from pysolar.solar import get_altitude
from datetime import timezone, timedelta, datetime
import numpy as np
import logging
loglevel = logging.INFO
logfmt = '[%(levelname)s] %(asctime)s - %(message)s'
logging.basicConfig(level=loglevel, format=logfmt)


def insert_into_dataset(pyroCb_id, piece_id, band, date_idx, success, opts):
    client = bigquery.Client()

    query = "UPDATE {}.fires".format(opts.bigquery)

    query += " SET status = '{}'".format(success)

    clause = "WHERE piece_id > 0"

    clause += " AND pyroCb_id = {} ".format(pyroCb_id)
    clause += " AND piece_id = {} ".format(piece_id)
    clause += " AND band = {} ".format(band)
    clause += " AND date_idx = {} ".format(date_idx)

    sort_clause = """
        ORDER BY 
            pyroCb_id, 
            piece_id, 
            date_idx,
            band,
            status
                  """

    query = query + clause + sort_clause

    query_job = client.query(query)  # API request
    # query_job.result()  # Waits for statement to finish


def create_dataset(dataset_id, region_name):
    """Create a dataset in Google BigQuery if it does not exist.

    :param dataset_id: Name of dataset
    :param region_name: Region name for data center, i.e. europe-west2 for London
    :return: Create dataset
    """

    client = bigquery.Client()
    reference = client.dataset(dataset_id)

    try:
        client.get_dataset(reference)
    except NotFound:
        dataset = bigquery.Dataset(reference)
        dataset.location = region_name

        dataset = client.create_dataset(dataset)


def insert_dataset(df, table):
    """Insert data from a Pandas dataframe into Google BigQuery. 
    :param df: Name of Pandas dataframe
    :param table: Name of BigQuery dataset and table, i.e. competitors.products
    :return: BigQuery job object"""

    client = bigquery.Client()
    return client.load_table_from_dataframe(df, table)


def delete_dataset(dataset):
    """This code deletes a big query data set
    : param dataset: Name of big query data set
    """
    client = bigquery.Client()
    try:
        client.delete_table(dataset)
    except NotFound:
        print("Dataset is not found.")

# this figures out what UTC time we should start at for the satellite images


def getlocalmidnightutc(currtimezone, currdate):
    # currtimezone: current time zone, e.g. 'America/Anchorage'
    # currdate: a string that gives the date as "YYYY-MM-DD"
    # return the local midnight date/time given the timezone as UTC date/time ('%Y-%m-%d %H:%M:%S %Z%z')
    localtz = pytz.timezone(currtimezone)
    utcdate = currdate
    yr, mo, day = utcdate.split("-")

    loc_dt = localtz.localize(datetime(int(yr), int(mo), int(day), 0, 0, 0))
    utc = pytz.utc
    return str(loc_dt.astimezone(utc))

# calculates datetime sequence to download from start_datetime, end_datetime and frequency (in minutes)
# start_datetime and end_datetime are assumed to have MM:SS=00:00


def getDateTimes_for_wildfire_piece(ini_time, size_days, freq):
    end_time = ini_time + timedelta(days=size_days)
    return [(ini_time + timedelta(hours=freq * i / 60)) for i in range(int((end_time-ini_time).total_seconds()/60. / freq))]


def solar_zenith_angle(lon: float, lat: float, date_time: datetime) -> float:
    """
    Calculates solar-zenith-angel

    Args:
        lon (float): _description_
        lat (float): _description_
        date_time (float): _description_

    Returns:
        (float): solar zenith angle
    """
    dobj = date_time.replace(tzinfo=timezone.utc)
    sza = float(90) - get_altitude(lat, lon, dobj)
    return sza


def filter_daytime_dates(lat, lon, dates_extract, thrs):
    """
    Filters datetime sequence based on solar-zenith-angle

    Args:
        lat (_type_): _description_
        lon (_type_): _description_
        dates_extract (_type_): _description_
        thrs (_type_): _description_

    Returns:
        _type_: _description_
    """
    szas = [solar_zenith_angle(lat, lon, dates_extract[i])
            for i in range(len(dates_extract))]
    indx, = np.where(np.array(szas) < thrs)
    return [dates_extract[i] for i in indx]


def getDatePieces(df, i, size_days):
    ini = df.iloc[i].extract_ini_date_utc
    num_pieces = int(np.ceil(df.iloc[i].length_fire / size_days))
    days_add = np.linspace(0, size_days * (num_pieces), num_pieces + 1)
    indx = np.where(days_add < df.iloc[i].length_fire)
    days_add = days_add[indx]
    date_list_ini = [ini + timedelta(days=x) for x in days_add.tolist()]
    date_list_ini

    return date_list_ini


def checkdate(curryear, currmonth, currday, opyear, opmonth, opday):
    # check whether the date is before or after the satellite became operational
    # currday,currmonth,currday - the date we want
    # opyear,opmonth,opday - the date the satellite became operational

    if curryear > opyear:
        return True
    elif curryear == opyear:
        if currmonth > opmonth:
            return True
        elif currmonth == opmonth:
            if currday >= opday:
                return True
            else:
                return False
    else:
        return False


def checkdatetime(curryear, currmonth, currday, currhour, opyear, opmonth, opday, ophour):
    # check whether the date is before or after the satellite became operational
    # currday,currmonth,currday - the date we want
    # opyear,opmonth,opday - the date the satellite became operational

    if curryear > opyear:
        return True
    elif curryear == opyear:
        if currmonth > opmonth:
            return True
        elif currmonth == opmonth:
            if currday >= opday:
                return True
            elif currday == opday:
                if currhour >= ophour:
                    return True
                else:
                    return False
    else:
        return False


def getsatellite(latitude, longitude, currdate):
    # figure out whether we would want himawari, goes-16, or goes-17
    utcdate = currdate
    yr, mo, dy = utcdate.split("-")
    year = float(yr)
    month = float(mo)
    day = float(dy)

    if longitude > 0:
        # Himawari data available from July 2015
        if checkdate(year, month, day, 2015, 7, 1):
            return "Himawari"
        else:
            return "NaN"
    else:
        # GOES-16 becomes operational Dec. 18 2017
        if longitude > -110:
            if checkdate(year, month, day, 2017, 12, 18):
                return "GOES16"
            else:
                return "NaN"

        # GOES-17 becomes operational Feb. 12 2019
        else:
            if checkdate(year, month, day, 2019, 2, 12):
                return "GOES17"
            elif checkdate(year, month, day, 2017, 12, 18):
                return "GOES16"
            else:
                return "NaN"


def getscanmode(currdatetime, satellite):
    # figure out whether we would want himawari, goes-16, or goes-17
    currdate = str(currdatetime).split()[0]
    currtime = str(currdatetime).split()[1]
    utcdate = currdate
    yr, mo, dy = utcdate.split("-")
    year = float(yr)
    month = float(mo)
    day = float(dy)
    hour = float(currtime.split(":")[0])

    if satellite == "Himawari":
        # Scan mode changed from 3 to 6 on 2019-04-02 16:00 UTC
        return None
    else:
        if checkdatetime(year, month, day, hour, 2019, 4, 2, 16):
            return 6
        else:
            return 3

# Function to "explode" wildfire location database into non-overlapping 1-day pieces (wildire_pieces)
# # and snapshots-by-band  (wildfire_snapshots)

# size_days: size of pieces


def blowUpWildfires(wildfires, size_days):

    wildfire_pieces = wildfires.copy()

    # housekeeping
    wildfire_pieces = wildfire_pieces[[
        'extract_longitude', 'extract_latitude', 'pyroCb_id', 'extract_ini_date_utc', 'extract_end_date_utc']]
    wildfire_pieces = wildfire_pieces.rename(
        columns={'extract_longitude': 'longitude', 'extract_latitude': 'latitude'})

    # blow up  into 1 day non-overlaping wildfire pieces
    wildfire_pieces["date_pieces"] = [getDatePieces(
        wildfires, i, size_days) for i in range(wildfires.shape[0])]
    ids = [np.linspace(1, len(wildfire_pieces.iloc[i]["date_pieces"]), len(
        wildfire_pieces.iloc[i]["date_pieces"]), dtype=int).tolist() for i in range(wildfires.shape[0])]
    ids = [ids[i][j] for i in range(len(ids)) for j in range(len(ids[i]))]
    wildfire_pieces = wildfire_pieces.explode('date_pieces')
    wildfire_pieces["wildfire_piece_id"] = [str(
        wildfire_pieces.iloc[i].pyroCb_id)+"_"+str(ids[i]) for i in range(wildfire_pieces.shape[0])]
    wildfire_pieces["piece_id"] = [ids[i]
                                   for i in range(wildfire_pieces.shape[0])]

    # housekeeping
    wildfire_pieces = wildfire_pieces.drop(
        columns=["extract_ini_date_utc", "extract_end_date_utc"])
    wildfire_pieces = wildfire_pieces.rename(
        columns={"date_pieces": "extract_ini_date"})
    wildfire_pieces = wildfire_pieces[[
        'pyroCb_id', 'piece_id', 'wildfire_piece_id', 'longitude', 'latitude', 'extract_ini_date']]

    # find correct satellite to query
    wildfire_pieces["satellite"] = [getsatellite(wildfire_pieces.iloc[i].latitude, wildfire_pieces.iloc[i].longitude, str(
        wildfire_pieces.iloc[i].extract_ini_date).split()[0]) for i in range(wildfire_pieces.shape[0])]

    return wildfire_pieces


# find closest available goes data before or after a certain date
def getIndx(type_dates, indx, i, maxLook, after):
    indx2, = np.where(
        type_dates[indx[i]+after*np.linspace(0, maxLook, maxLook+1, dtype=int)] == 1)
    return after*indx2[0]


# Get a desired-date to available-goes date conversion table for all "extractable"dates in
# wildfire_pieces dataframe
def getDatesKeys(wildfire_pieces, available_goes_dates, size_days, freq):
    min_ini_date = np.min(wildfire_pieces.extract_ini_date)
    max_ini_date = np.max(wildfire_pieces.extract_ini_date) + \
        timedelta(days=size_days)
    period_all = max_ini_date - min_ini_date
    dates_extract = getDateTimes_for_wildfire_piece(
        min_ini_date, period_all.days, freq)
    dates_extract = np.array(dates_extract)
    all_dates = np.hstack([available_goes_dates, dates_extract])
    type_dates = np.hstack(
        [np.ones(available_goes_dates.shape[0]), 2*np.ones(dates_extract.shape[0])])
    o = np.argsort(all_dates)
    all_dates = all_dates[o]
    type_dates = type_dates[o]
    indx, = np.where(type_dates == 2)
    indxBefore = np.array([getIndx(type_dates, indx, i, 10, -1)
                          for i in range(indx.shape[0])])
    indxAfter = np.array([getIndx(type_dates, indx, i, 10, 1)
                         for i in range(indx.shape[0])])
    dateBefore = all_dates[indx+indxBefore]
    dateAfter = all_dates[indx+indxAfter]
    distBefore = (dates_extract - dateBefore)
    distBefore = np.array([distBefore[i].total_seconds()
                          for i in range(distBefore.shape[0])])
    distAfter = (dateAfter-dates_extract)
    distAfter = np.array([distAfter[i].total_seconds()
                         for i in range(distAfter.shape[0])])
    dists = np.vstack([distBefore, distAfter]).T
    indx2 = np.vstack([indxBefore, indxAfter]).T
    indxWhich = np.apply_along_axis(np.argmin, 1, dists)
    indx3 = np.array([indx2[i, indxWhich[i]]
                     for i in range(indxWhich.shape[0])])
    closestGoesDate = all_dates[indx+indx3]
    dates_keys = np.vstack([dates_extract, closestGoesDate]).T
    return dates_keys

# find available date index for one date to extract


def findDateKey(dates_extract, dates_keys, j):
    indx, = np.where(dates_extract[j] == dates_keys[:, 0])
    return indx[0]

# constructs desired extract dates, filters by solar zenith angle and matches to available
# dates in case it corresponds to GOES satellite


def getDates(wildfire_pieces, i, size_days, freq, sza_thrs, dates_keys):
    if i % 300 == 0:
        logging.info("i: "+str(i))
    dates_extract = getDateTimes_for_wildfire_piece(
        wildfire_pieces.iloc[i].extract_ini_date, size_days, freq)
    dates_extract = filter_daytime_dates(
        wildfire_pieces.iloc[i].latitude, wildfire_pieces.iloc[i].longitude, dates_extract, sza_thrs)
    dates_extract = np.array(dates_extract)
    # print(dates_extract.shape)

    if wildfire_pieces.iloc[i].satellite != "Himawari":
        indx = np.array([findDateKey(dates_extract, dates_keys, j)
                        for j in range(dates_extract.shape[0])])
        # print(indx.shape)
        if indx.shape[0] == 0:
            dates_extract = None
        else:
            dates_extract = dates_keys[indx, 1]

    return dates_extract


# bands - bands to retrieve  dictionary by satllite
# sza_thrs solar_zenith angle to use to filter out dates to sample
# sampling frequency in minutes
def blowUpWildfirePieces(wildfire_pieces, bands, sza_thrs, size_days, freq, available_goes_dates):

    dates_keys = getDatesKeys(
        wildfire_pieces, available_goes_dates, size_days, freq)

    # blow up by snapshots
    wildfire_snapshots = wildfire_pieces.copy()
    wildfire_snapshots["snapshot"] = [getDates(
        wildfire_pieces, i, size_days, freq, sza_thrs, dates_keys) for i in range(wildfire_pieces.shape[0])]

    # delete pieces with no available dates
    idx = [np.all(np.logical_not(pd.isnull(wildfire_snapshots.iloc[i].snapshot)))
           for i in range(wildfire_snapshots.shape[0])]
    idx, = np.where(idx)
    print("keep: ", idx.shape[0], " out of: ", wildfire_pieces.shape[0])
    wildfire_snapshots = wildfire_snapshots.iloc[idx]

    date_idx = [np.linspace(1, len(wildfire_snapshots.iloc[i]["snapshot"]),
                            len(wildfire_snapshots.iloc[i]["snapshot"]), dtype=int).tolist() for i in range(wildfire_snapshots.shape[0])]
    date_idx = [date_idx[i][j]
                for i in range(len(date_idx)) for j in range(len(date_idx[i]))]

    wildfire_snapshots = wildfire_snapshots.explode('snapshot')
    wildfire_snapshots["date_idx"] = date_idx

    # find correct scanmode
    wildfire_snapshots["scanmode"] = [getscanmode(
        wildfire_snapshots.iloc[i].extract_ini_date, wildfire_snapshots.iloc[i].satellite) for i in range(wildfire_snapshots.shape[0])]

    #bands = [b+1 for b in range(num_bands)]

    # download status pending/success/fail
    wildfire_snapshots["status"] = "pending"
    alaska_locations_outofview = [258, 259, 260, 265, 266]
    wildfire_snapshots["status"].loc[wildfire_snapshots.pyroCb_id.isin(
        alaska_locations_outofview)] = "outofview"

    # blow up by bands
    wildfire_snapshots["band"] = [bands[wildfire_snapshots.iloc[i].satellite]
                                  for i in range(wildfire_snapshots.shape[0])]
    wildfire_snapshots = wildfire_snapshots.explode('band')

    # housekeeping
    wildfire_snapshots = wildfire_snapshots.drop(columns="extract_ini_date")

    return wildfire_snapshots
