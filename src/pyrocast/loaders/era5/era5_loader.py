
# These script requires the use of Copernicus Data Store API
#  1. Download the python library using 'pip install cdsapi'
#  2. Register yourself on the Copernicus Data Store
#  3. Set up the key following the instructions on the website
#
#


import cdsapi
from os import system

# Setup client
c = cdsapi.Client()


# Spatio-temporal variables

years = ['2019']

months = ['01', '02', '03',
          '04', '05', '06',
          '07', '08', '09',
          '10', '11', '12']

days = ['01', '02', '03'
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31']

times = ['00:00', '01:00', '02:00',
         '03:00', '04:00', '05:00',
         '06:00', '07:00', '08:00',
         '09:00', '10:00', '11:00',
         '12:00', '13:00', '14:00',
         '15:00', '16:00', '17:00',
         '18:00', '19:00', '20:00',
         '21:00', '22:00', '23:00']

areas = ['north_america', ]  # 'australia']
area_dict = {'north_america': [70, -170, 25, -50],
             'australia': [-10, 110, -45, 155]}


# Pressure variables

pressure_variables = ['relative_humidity', 'vertical_velocity',
                      'u_component_of_wind', 'v_component_of_wind']

pressure_dict = {'relative_humidity': ['650', '750', '850'],
                 'u_component_of_wind': '250',
                 'v_component_of_wind': '250',
                 'vertical_velocity': '1'}


# Single level variables

single_lvl_variables = ['convective_inhibition', 'geopotential',
                        'surface_latent_heat_flux', 'surface_sensible_heat_flux',
                        '10m_u_component_of_wind', '10m_v_component_of_wind',
                        '10m_wind_gust_since_previous_post_processing', 'boundary_layer_height',
                        'convective_available_potential_energy']


# Fuel variables
fuel_variables = ['high_vegetation_cover', 'low_vegetation_cover',
                  'type_of_high_vegetation', 'type_of_low_vegetation']


def download_pressure_variables(years: list, months: list, days: list, times: list, areas: list,
                                area_dict: dict, pressure_variables: list, pressure_dict: dict):
    """
    Download pressure variables and copies data to Google Cloud Storage Bucket.

    Args:
        years (list): list of years (strings) to download
        months (list): list of months (strings) to download
        days (list): list of days (strings) to download
        times (list): list of times (strings) to download
        areas (list): list of areas to dowload
        area_dict (dict): dictionnary defining area extents using lat/lon coordinates
        pressure_variables (list): list of pressure variables to dowload
        pressure_dict (dict): dictionnary defining pressure levels to download for each pressure variable
    """

    for y in years:
        for v in pressure_variables:
            for a in areas:
                filename = v + '_' + y + '.nc'

                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': v,
                        'pressure_level': pressure_dict[v],
                        'year': y,
                        'month': months,
                        'day': days,
                        'time': times,
                        'area': area_dict[a],
                    },
                    filename)

                move_command = 'gsutil cp ' + filename + \
                    ' gs://eu-aerosols-landing/ERA5/raw_data/' + a + '/'
                system(move_command)

                # delete file once it's been copied
                del_command = 'rm ' + filename
                system(del_command)


def download_single_lvl_variables(years: list, months: list, days: list, times: list,
                                  areas: list, area_dict: dict, single_lvl_variables: list):
    """
    Download single level variables and copies data to Google Cloud Storage Bucket.

    Args:
        years (list): list of years (strings) to download
        months (list): list of months (strings) to download
        days (list): list of days (strings) to download
        times (list): list of times (strings) to download
        areas (list): list of areas to download
        area_dict (dict): dictionnary defining area extents using lat/lon coordinates
        single_lvl_variables (list): list of single level variables to download
    """
    for y in years:
        for v in single_lvl_variables:
            for a in areas:

                filename = v + '_' + y + '.nc'

                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': v,
                        'year': y,
                        'month': months,
                        'day': days,
                        'time': times,
                        'area': area_dict[a],
                    },
                    filename)

                move_command = 'gsutil cp ' + filename + \
                    ' gs://eu-aerosols-landing/ERA5/raw_data/' + a + '/'
                system(move_command)

                # delete file once it's been copied
                del_command = 'rm ' + filename
                system(del_command)


def download_single_lvl_variables(years: list, months: list, days: list, times: list,
                                  areas: list, area_dict: dict, fuel_variables: list):
    """
    Download single level variables and copies data to Google Cloud Storage Bucket.

    Args:
        years (list): list of years (strings) to download
        months (list): list of months (strings) to download
        days (list): list of days (strings) to download
        times (list): list of times (strings) to download
        areas (list): list of areas to download
        area_dict (dict): dictionnary defining area extents using lat/lon coordinates
        single_lvl_variables (list): list of single level variables to download
    """
    for y in years:
        for a in areas:

            filename = 'fuel_' + y + '.nc'

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': fuel_variables,
                    'year': y,
                    'month': months,
                    'day': days,
                    'time': times,
                    'area': area_dict[a],
                },
                filename)

            move_command = 'gsutil cp ' + filename + \
                ' gs://eu-aerosols-landing/ERA5/raw_data/' + a + '/'
            system(move_command)

            # delete file once it's been copied
            del_command = 'rm ' + filename
            system(del_command)
