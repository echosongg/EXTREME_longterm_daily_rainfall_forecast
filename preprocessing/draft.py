import os
import xarray as xr
from preprocessing_utils import select_data, resize_data
import numpy as np
import pandas as pd
import cv2


# Define the ensemble models and directories
ensemble = ["e01"]
print("ensemble",ensemble)
input_directory = "/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/"
output_directory = "../"

def main():
    # Iterate through each ensemble member
    num = 0
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
            print("target_path", target_path)
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
                print(time_value)
                num =  num + 1
                break

            # Merge the DataArrays into a single dataset
            ds_total = xr.concat(ds_total, dim='time')

            # Save and close
            ds_total.to_netcdf(target_path)
            ds_raw.close()
            if num == 5:
                break


if __name__ == "__main__":
    # If the output directory does not exist, create it
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for exx in ensemble:
        if not os.path.exists(os.path.join(output_directory, exx)):
            os.mkdir(os.path.join(output_directory, exx))

    # Preprocess the data
    main()



'''# 假设你有一个 ACCESS 数据文件的路径
access_data_file = '/scratch/iu60/xs5813/Awap_pre_data/1988-05-28.nc'  # 替换为你的文件路径

# 使用 xarray 打开数据文件
ds = xr.open_dataset(access_data_file)
    
var = ds['pr'].values
print(f"AWAP data stats - Max: {np.max(var)}, Min: {np.min(var)}")

# 打印数据集的概要信息
print(ds)

# 打印数据集的维度
print("\nDimensions:")
print(ds.dims)

# 打印数据集中的变量
print("\nVariables:")
print(ds.data_vars)

# 打印数据集的属性
print("\nAttributes:")
print(ds.attrs)
'''