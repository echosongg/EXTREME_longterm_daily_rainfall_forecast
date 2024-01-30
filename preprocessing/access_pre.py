import os
import xarray as xr
from preprocessing_utils import select_data, resize_data
import numpy as np
import pandas as pd
import cv2


# Define the ensemble models and directories
ensemble = ["e02","e03"]
print("ensemble",ensemble)
input_directory = "/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/"
output_directory = "/scratch/iu60/xs5813/Processed_data/"

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
                da_raw = ds_raw.sel(time=time_value)['pr'] * 86400 # Convert from kg m-2 s-1 to mm/day
                
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

'''
ACCESS的structure:
The pr's cooridinate is lon, lat and time.
Time in original file means initial time, and it will contain many other time(200+)
'''

