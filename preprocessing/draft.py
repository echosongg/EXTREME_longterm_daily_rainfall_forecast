import os
import xarray as xr
import datetime

# Set the directory where your .nc files are located
data_directory = '/g/data/zv2/agcd/v1/precip/total/r005/01day'

times = 0
# Loop through each file in the directory
for filename in os.listdir(data_directory):
    # Check if the file is a NetCDF file
    # Construct the full path to the file and open it as an xarray dataset
    file_path = os.path.join(data_directory, filename)
    ds_raw = xr.open_dataset(file_path)
    times = times + 1
    # Process each time value in the dataset
    for time_value in ds_raw['time'].values:
        # Extract the leading year and month from the time value
        leading_year = str(datetime.datetime.utcfromtimestamp(time_value.astype(int) * 1e-9).year)
        leading_month = str(datetime.datetime.utcfromtimestamp(time_value.astype(int) * 1e-9).month).zfill(2)
        leading_day = str(datetime.datetime.utcfromtimestamp(time_value.astype(int) * 1e-9).day).zfill(2)
        print(leading_year, leading_month, leading_day)
    if times == 100:
        break

