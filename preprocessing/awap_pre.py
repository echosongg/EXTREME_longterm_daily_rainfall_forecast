import os
import xarray as xr
from preprocessing_utils import select_data
import datetime
import numpy as np

# Define input and output directories
data_directory = "/g/data/zv2/agcd/v1/precip/total/r005/01day"
output_directory = "/scratch/iu60/xs5813/Awap_pre_data/"

def get_days_in_month(date:str) -> int:
    """
    Returns the number of days in the specified month.

    Parameters:
    - date (str): The date in the format 'YYYY-MM'.                 

    Returns:
    - int: The number of days in the specified month.
    """
    def year_is_leap(year: str):
                year = int(year)
                return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    year, month = date.split('-')
    if month in ['01', '03', '05', '07', '08', '10', '12']:
        return 31
    elif month in ['04', '06', '09', '11']:
        return 30
    elif year_is_leap(year):
        return 29
    else:
        return 28 

def main():

    # Iterate through each file in the data directory
    for filename in os.listdir(data_directory):

        # Construct the full path to the file and open it as an xarray dataset
        file_path = os.path.join(data_directory, filename)
        ds_raw = xr.open_dataset(file_path)

        # Process each time value in the dataset
        for time_value in ds_raw['time'].values:
            # Extract the leading year and month from the time value
            leading_year = str(datetime.datetime.utcfromtimestamp(time_value.astype(int) * 1e-9).year)
            leading_month = str(datetime.datetime.utcfromtimestamp(time_value.astype(int) * 1e-9).month).zfill(2)
            leading_day = str(datetime.datetime.utcfromtimestamp(time_value.astype(int) * 1e-9).day).zfill(2)

            
            # Select the 'precip' data for the current time and apply geographic selection and resizing
            da_raw = ds_raw.sel(time=time_value)['precip'] #/ get_days_in_month(f"{leading_year}-{leading_month}") # Average daily rainfall
            da_selected = select_data(da_raw)

       

            da_to_save = xr.DataArray(da_selected, dims=("lat", "lon"),
                                    coords={
                                        "lat": da_selected.lat, 
                                        "lon": da_selected.lon
                                    }, name='pr')
            
                        
            da_to_save.to_netcdf(os.path.join(output_directory, f"{leading_year}-{leading_month}-{leading_day}.nc"))

        # Close the dataset to free resources
        ds_raw.close()


if __name__ == "__main__":
    # If the output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Preprocess the data
    main()
