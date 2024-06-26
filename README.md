# Pyrocast

![Pyrocast logo](figures/pyrocast_logo_colour.png)

## Introduction

Pyrocast is a end-to-end machine learning pipeline for the prediction of extreme and dangerous wildfires. More specifically the code in this repository allows you to find, forecast and understand the causal drivers of pyrocumolonimbus clouds, precursors the most large and unpredictable wildfires. The pipeline includes:

* `loaders` to download and format the data
* `nrl_algorithm` to find pyrocumolonimbus clouds and label the data
* `models` to forecast the pyrocumolonimbus clouds
* `icp` to understand the causal drivers of the pyrocumolonimbus clouds

The code for this repository is currently incomplete, the authors are contributing to the repository in their spare time so please be patient.

## Getting started

Get in touch with jodie@fdl.ai to get access to the data on [Google Cloud Storage](https://console.cloud.google.com/storage/browser/eu-aerosols-spaceml).


## Data

The data is in a [Zarr format](https://zarr.readthedocs.io/en/stable/), this allows us to load data that is associated to each hour of each day of each wildfire event using the ID numbers found in the `wildfire_events.csv` file. The data for the geostationary imagery, the pyrocast flags and masks and fuel and weather data each have their own Zarr directory.

Extracting data from a zarr folder event will yield Nx200x200 cube where N corresponds to the different wavelength channels, climate fields, etc.. These are detailed in the tables below.

### Flags and masks

PyroCb_flags.zarr (array shape = 1)
| N | Content |
|---|---|
| 0 | PyroCb flag, whether or not scene contains PyroCb |

PyroCb_mask.zarr (array shape = 1 x 200 x 200)
| N | Content |
|---|---|
| 0 | PyroCb mask, classification of pixel types according to NRL PyroCb algorithm |

### Geostationary imagery

Array shape = 18 x 200 x 200

Only some dimensions have entries which depend on the satellite source.

Himawari-8 
| N | Channel wavelength [μm]|
|---|---|
| 0| 0.47| 
| 2| 0.64|
| 3| 0.86|
| 6|  3.9|
|13| 11.2| 
|15| 13.3| 

GOES-16 / GOES-17
| N | Channel wavelength [μm]|
|---|---|
| 0| 0.47| 
| 1| 0.64|
| 2| 0.86|
| 6|  3.9|
|13| 11.2| 
|15| 13.3| 

### Weather and fuel

Array shape = 19 x 200 x 200

| N | Content |
|---|---|
| 1| 10m v component of wind |
| 2| 10m wind gust since previous post processing|
| 3| boundary layer height|
| 4| convective available potential energy|
| 5| convective inhibition|
| 6| geopotential|
| 7| surface latent heat flux
| 8| surface sensible heat flux|
| 9| surface vertical velocity |
|10| component of wind at 250hPa |
|11| v component of wind at 250hPa |
|12| fraction of high vegetation |
|13| fraction of low vegetation |
|14| type of high vegetation|
|15| type of low vegetation |
|16| relative humidity at 650hPa|
|17| relative humidity at 750hPa|
|18| relative humidity at 850hPa|
