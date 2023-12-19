import numpy as np
import xarray as xr
import cv2

"""
    Latitude and longitude ranges for case region (Australia).
"""
#经度范围：大约从东经 138 度到东经 153 度。
#纬度范围：大约从南纬 29 度到南纬 10.5 度。
#lon_range = (111.975,  156.275)
#lat_range = (-44.525, -9.975)
lon_range = (138,  156.275)
lat_range = (-29, -9.975)
scale = 1.5



def select_data(da, lon_range=lon_range, lat_range=lat_range):
    """
    Selects a subset of data based on the specified longitude and latitude ranges.

    Parameters:
    - da (xarray.DataArray): The data array to be selectolled from.
    - lon_range (tuple): The longitude range as (min_lon, max_lon).
    - lat_range (tuple): The latitude range as (min_lat, max_lat).

    Returns:
    - xarray.DataArray: The selected subset of the data array.
    """

    # Create boolean masks for the longitude and latitude ranges
    lon_mask = np.logical_and(da.lon >= lon_range[0], da.lon <= lon_range[1])
    lat_mask = np.logical_and(da.lat >= lat_range[0], da.lat <= lat_range[1])

    # Apply the masks and return
    return da.sel(lon=lon_mask, lat=lat_mask)








