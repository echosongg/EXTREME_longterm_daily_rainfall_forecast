import os
import xarray as xr
from preprocessing_utils import select_data
import datetime
import numpy as np

# Define input and output directories
data_directory = "/g/data/zv2/agcd/v1/precip/total/r005/01day"
output_directory = "/scratch/iu60/xs5813/draftresult/"

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


target_date = "1994-08-16"  # Set the target date here

def main():
    # Iterate through each file in the data directory
    for filename in os.listdir(data_directory):
        # Construct the full path to the file and open it as an xarray dataset
        file_path = os.path.join(data_directory, filename)
        ds_raw = xr.open_dataset(file_path)

        # Process each time value in the dataset
        for time_value in ds_raw['time'].values:
            # Extract the date string from the time value
            date_str = str(np.datetime_as_string(time_value, unit='D'))
            print(time_value)

            # Continue only if the date matches the target date
            if date_str == target_date:
                # Select the 'precip' data for the current time and apply geographic selection
                da_raw = ds_raw.sel(time=time_value)['precip']
                print("da_raw statistics:")
                print("Mean: ", da_raw.mean().values)
                print("Max: ", da_raw.max().values)
                print("Min: ", da_raw.min().values)
                print("Std: ", da_raw.std().values)

                da_selected = select_data(da_raw)
                print("after select lon and lat, the awap shape is", da_selected.shape)

                da_to_save = xr.DataArray(da_selected, dims=("lat", "lon"),
                                        coords={
                                            "lat": da_selected.lat, 
                                            "lon": da_selected.lon
                                        }, name='pr')

                da_to_save.to_netcdf(os.path.join(output_directory, f"{date_str}.nc"))
                break  # Once the target date is processed, break the loop

        # Close the dataset to free resources
        ds_raw.close()
        break  # Exit after processing the target date file

if __name__ == "__main__":
    print("awap data")
    # If the output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Preprocess the data
    main()
