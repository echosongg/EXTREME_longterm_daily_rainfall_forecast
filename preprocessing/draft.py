import os
import xarray as xr
import numpy as np
import pandas as pd
import cv2


"""
    Latitude and longitude ranges for case region (Australia).
"""
#经度范围：大约从东经 138 度到东经 153 度。
#纬度范围：大约从南纬 29 度到南纬 10.5 度。
#lon_range = (111.975,  156.275)
#lat_range = (-44.525, -9.975)
lon_range = (142.975,  153.765)
lat_range = (-31.525, -24.375)
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
    print("原始经度值: ", da.lon.values)
    print("原始纬度值: ", da.lat.values)
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
    print("data.lat.size", data.lat.size)
    print("data.lon.size", data.lon.size)
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
# Define the ensemble models and directories
ensemble = ["e01"]
print("ensemble",ensemble)
input_directory = "/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/"
output_directory = "/scratch/iu60/xs5813/draftresult/"
target_year = "1994"
target_date = "1994-08-16"  # 设置目标日期

def main():
    # Iterate through each ensemble member
    for exx in ensemble:
        exx_directory = os.path.join(input_directory, exx) # Path + ensemble member

        # Iterate through each file within the ensemble member's directory
        for filename in os.listdir(exx_directory):
            # Check if the file is for the target year
            if target_year in filename:
                # Construct the full path for the file
                file_path = os.path.join(exx_directory, filename)

                # Open the dataset
                ds_raw = xr.open_dataset(file_path)
                ds_raw = ds_raw.fillna(0) # Fill missing values with zero

                # Process each time value in the dataset
                for time_value in ds_raw['time'].values:
                    # Convert time to string and check if it matches the target date
                    time_str = str(np.datetime_as_string(time_value, unit='D'))
                    if time_str == target_date:
                        # Select the 'pr' data for the current time and apply geographic selection and resizing
                        da_raw = ds_raw.sel(time=time_value)['pr'] * 86400 # Convert from kg m-2 s-1 to mm/day

                        # Print statistics of da_raw
                        print("da_raw statistics for", time_value)
                        print("Mean: ", da_raw.mean().values)
                        print("Max: ", da_raw.max().values)
                        print("Min: ", da_raw.min().values)
                        print("Std: ", da_raw.std().values)
                
                        da_selected = select_data(da_raw)
                        da_to_save = resize_data(da_selected, np.datetime64(time_value))
                        
                        # Save the data
                        target_path = os.path.join(output_directory, exx, f"{time_str}.nc")
                        print("target path", target_path)
                        da_to_save.to_netcdf(target_path)

                # Close the dataset to free resources
                ds_raw.close()
                break  


if __name__ == "__main__":
    # If the output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for exx in ensemble:
        if not os.path.exists(os.path.join(output_directory, exx)):
            os.mkdir(os.path.join(output_directory, exx))

    # Preprocess the data
    print("AWAP data")
    main()

'''
ACCESS的structure:
The pr's cooridinate is lon, lat and time.
Time in original file means initial time, and it will contain many other time(200+)
'''
#1994-08-16T12:00:00.000000000
