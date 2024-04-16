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

Extracting data from a zarr folder event will yield Nx200x200 cube where N corresponds to the different wavelength channels, climate fields, etc. These are detailed in the tables below.
