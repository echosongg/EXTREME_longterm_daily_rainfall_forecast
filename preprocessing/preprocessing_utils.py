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
lon_range = (142.975,  154.275)
lat_range = (-31.525, -23.975)
#lon_range = (138,  156.275)
#lat_range = (-29, -9.975)
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

def resize_data(data, time_value):
    """
    Resizes the data array by a specified scale factor using cubic interpolation.

    Parameters:
    - data (xarray.DataArray): The data array to be resized.
    - time_value (np.datetime64): The time value of the data array.

    Returns:
    - xarray.DataArray: The resized data array.
    """
    # Resize the data array using OpenCV
    #new_shape = (86, 110) # Shift to same proportion as AWAP + 1.5x scale  60->40km???
    new_shape = (int(data.lat.size * scale), int(data.lon.size * scale))

    # Resize using cubic interpolation (cv2 treats the shape as (width, height) instead of (height, width) hence the [::-1])
    resized_values = cv2.resize(data.values, new_shape[::-1], interpolation=cv2.INTER_CUBIC) # todo - is this the right interpolation method? 
    resized_values = np.clip(resized_values, 0, None)  # Clipping negative values

    # Create new longitude and latitude arrays based on the new shape
    new_lon = np.linspace(data.lon[0], data.lon[-1], new_shape[1]) 
    new_lat = np.linspace(data.lat[0], data.lat[-1], new_shape[0]) 

    coords = {
        "lat": new_lat,
        "lon": new_lon,
        #"time": np.datetime_as_string(time_value, unit='D')[:-3] + "-01" # YYYY-MM-DD
        "time": np.datetime_as_string(time_value, unit='D') 
    }

    # Create a new DataArray and return
    return xr.DataArray(resized_values, dims=("lat", "lon"), coords=coords, name='pr') 







