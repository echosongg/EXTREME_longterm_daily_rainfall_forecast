import os
import xarray as xr
from preprocessing_utils import select_data
import numpy as np
import pandas as pd
import cv2


# Define the ensemble models and directories
ensemble = ["e09"]
input_directory = "/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/"
output_directory = "/scratch/iu60/xs5813/Processed_data/"

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
    new_shape = (86, 110) # Shift to same proportion as AWAP + 1.5x scale  60->40km???
    # Resize using cubic interpolation (cv2 treats the shape as (width, height) instead of (height, width) hence the [::-1])
    resized_values = cv2.resize(data.values, new_shape[::-1], interpolation=cv2.INTER_CUBIC) # todo - is this the right interpolation method? 
    resized_values = np.clip(resized_values, 0, None)  # Clipping negative values

    # Create new longitude and latitude arrays based on the new shape
    new_lon = np.linspace(data.lon[0], data.lon[-1], new_shape[1]) 
    new_lat = np.linspace(data.lat[0], data.lat[-1], new_shape[0]) 

    coords = {
        "lat": new_lat,
        "lon": new_lon,
        "time": np.datetime_as_string(time_value, unit='D')[:-3] + "-01" # YYYY-MM-DD
    }

    # Create a new DataArray and return
    return xr.DataArray(resized_values, dims=("lat", "lon"), coords=coords, name='pr') 

def main():
    # Iterate through each ensemble member
    for exx in ensemble:
        exx_directory = os.path.join(input_directory, exx) # Path + ensemble member

        # Iterate through each file within the ensemble member's directory
        for filename in os.listdir(exx_directory):
            #if not (1990 <= int(filename[6:10])):
            #    continue
            # Construct the full path for the file
            file_path = os.path.join(exx_directory, filename)

            # Rename file e.g: ma_pr_19960623_e01.nc -> 1996-06-23.nc
            new_filename = f"{filename[6:10]}-{filename[10:12]}-{filename[12:14]}.nc"
            target_path = os.path.join(output_directory, exx, new_filename) 
            # Skip processing if the file already exists
            # if os.path.exists(target_path):
            #     continue

            # Open the dataset
            ds_raw = xr.open_dataset(file_path)
            ds_raw = ds_raw.fillna(0) # Fill missing values with zero

            # Process each time value in the dataset
            ds_total = []
            for time_value in ds_raw['time'].values:
                # Select the 'pr' data for the current time and apply geographic selection and resizing
                da_raw = ds_raw.sel(time=time_value)['pr'] * 86000 # Convert from kg m-2 s-1 to mm/day
                
                da_selected = select_data(da_raw)
                da_to_save = resize_data(da_selected, time_value)
                                
                ds_total.append(da_to_save)

            # Merge the DataArrays into a single dataset
            ds_total = xr.concat(ds_total, dim='time')

            # Save and close
            ds_total.to_netcdf(target_path)
            ds_raw.close()


if __name__ == "__main__":
    # If the output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for exx in ensemble:
        if not os.path.exists(os.path.join(output_directory, exx)):
            os.mkdir(os.path.join(output_directory, exx))

    # Preprocess the data
    main()

