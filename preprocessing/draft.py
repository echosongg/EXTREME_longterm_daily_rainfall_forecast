import os
import xarray as xr
import datetime

# Set the directory where your .nc files are located
data_directory = '/scratch/iu60/xs5813/Awap_pre_data/'

import xarray as xr

# 打开 NetCDF 文件
ds = xr.open_dataset('/scratch/iu60/xs5813/Awap_pre_data/1946-03-23.nc')

# 打印出文件中所有的变量名
print(ds.variables)

# 检查 'precip' 是否在变量名中
if 'precip' in ds.variables:
    print("Found 'precip'")
else:
    print("'precip' not found in the dataset")

