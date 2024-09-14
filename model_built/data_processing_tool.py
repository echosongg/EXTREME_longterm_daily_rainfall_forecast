import cv2
import xarray as xr
from netCDF4 import Dataset, num2date, date2num
# from libtiff import TIFF
import os, sys
import numpy as np
from datetime import date, timedelta, datetime
import random
import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
# import cartopy.crs as ccrs
from matplotlib import cm
# from mpl_toolkits.basemap import Basemap
import warnings

# warnings.filterwarnings("ignore")
#
# levels = {}
# levels["hour"] = [0., 0.2, 1, 5, 10, 20, 30, 40, 60, 80, 100, 150]
# levels["day"] = [0., 0.2, 5, 10, 20, 30, 40, 60, 100, 150, 200, 300]
# levels["week"] = [0., 0.2, 10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]
# levels["month"] = [0., 10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
# levels["year"] = [0., 50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]
# enum = {0: "0600", 1: "1200", 2: "1800", 3: "0000", 4: "0600"}
#
# prcp_colours_0 = [
#     "#FFFFFF",
#     '#ffffd9',
#     '#edf8b1',
#     '#c7e9b4',
#     '#7fcdbb',
#     '#41b6c4',
#     '#1d91c0',
#     '#225ea8',
#     '#253494',
#     '#081d58',
#     "#4B0082"]
#
# prcp_colours = [
#     "#FFFFFF",
#     '#edf8b1',
#     '#c7e9b4',
#     '#7fcdbb',
#     '#41b6c4',
#     '#1d91c0',
#     '#225ea8',
#     '#253494',
#     '#4B0082',
#     "#800080",
#     '#8B0000']
#
# prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)


def read_awap_data_fc(root_dir, date_time):
    # filename = root_dir + (date_time + timedelta(1)).strftime("%Y%m%d") + ".nc"
    filename = root_dir + (date_time).strftime("%Y-%m-%d") + ".nc"
    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)
    #     print(data)# lat(324), lon(432)
    var = dataset.isel(time=0)['precip'].values
    var = np.squeeze(var)
    dataset.close()
    return var, date_time


def read_awap_data_fc_get_lat_lon(root_dir, date_time):  # precip_calib_0.05_1911
    # filename=root_dir+(date_time+timedelta(1)).strftime("%Y%m%d")+".nc"
    filename = root_dir + (date_time).strftime("%Y-%m-%d") + ".nc"
    data = Dataset(filename, 'r')
    lats = data['lat'][:]
    lons = data['lon'][:]
    var = data["precip"][:]
    var = var.filled(fill_value=0)
    var = np.squeeze(var)
    data.close()
    return var, lats, lons


def read_access_data(root_dir, en, date_time, leading, var_name="pr"):
    filename = root_dir + en + "/" + date_time.strftime("%Y-%m-%d") + "_" + en + ".nc"
    data = Dataset(filename, 'r')
    var = data[var_name][leading]
    var = var.filled(fill_value=0)
    # var = cv2.resize(var, dsize=(886, 691), interpolation=cv2.INTER_CUBIC)
    data.close()
    return var


def read_access_data_calibration(root_dir, en, date_time, leading, year, var_name=["pr","alpha","beta"]):
    #filename = root_dir + en + "/" + var_name + "/" year +"/" + date_time.strftime("%Y-%m-%d") + ".nc"
    filename = root_dir + en + "/" + var_name + "/" + year + "/" + date_time.strftime("%Y-%m-%d") + ".nc"
    #/scratch/iu60/xs5813/TestResults/DESRGAN/vTestRefactored/model_G_i000005_20240403-035316_with_huber
    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)
    #print("access filename",filename)
    # Check if 'leading' index is within the bounds of the 'time' dimension
    if leading >= len(dataset['time']):
        raise IndexError(f"Index {leading} is out of bounds for axis 'time' with size {len(dataset['time'])}")
    var = dataset.isel(time=leading)[var_name].values
    #print("access var:", var)
    dataset.close()
    return var

def read_access_data_calibration(root_dir, en, date_time, leading, year, var_name="pr"):
    if isinstance(var_name, str):
        # Handle single variable name (e.g., "pr")
        filename = f"{root_dir}{en}/{var_name}/{year}/{date_time.strftime('%Y-%m-%d')}.nc"
        dataset = xr.open_dataset(filename)
        #print("access filename:", filename)
    else:
        # Handle list of variable names (e.g., ["p", "alpha", "beta"])
        datasets = []
        for var in var_name:
            filename = f"{root_dir}{en}/{var}/{year}/{date_time.strftime('%Y-%m-%d')}.nc"
            ds = xr.open_dataset(filename)
            datasets.append(ds[var].expand_dims('variable'))
            #print("access filename:", filename)
        dataset = xr.concat(datasets, dim='variable')
        dataset['variable'] = var_name

    dataset = dataset.fillna(0)
    # Select the data at the 'leading' time index
    if isinstance(var_name, str):
        var = dataset.isel(time=leading)[var_name]
    else:
        var = dataset.isel(time=leading)
    dataset.close()
    return var

def read_access_data_calibrataion_get_lat_lon(root_dir, en, date_time, leading, var_name="pr"):
    filename = root_dir + en + "/" + date_time.strftime("%Y-%m-%d") + "_" + en + ".nc"
    data = Dataset(filename, 'r')
    var = data[var_name][leading]
    var = var.filled(fill_value=0)
    lats = data['lat'][:]
    lons = data['lon'][:]
    data.close()
    return var, lats, lons


def read_access_data_get_lat_lon(root_dir, en, date_time, leading, var_name="pr"):
    filename = root_dir + en + "/" + date_time.strftime("%Y-%m-%d") + "_" + en + ".nc"
    data = Dataset(filename, 'r')
    var = data[var_name][leading]
    var = var.filled(fill_value=0)
    lats = data['lat'][:]
    lons = data['lon'][:]
    data.close()
    return var, lats, lons


def read_access_data_get_lat_lon_30(root_dir, en, date_time, leading, var_name="pr"):
    filename = root_dir + en + "/" + date_time.strftime("%Y-%m-%d") + "_" + en + ".nc"
    data = Dataset(filename, 'r')
    var = data[var_name][leading]
    var = var.filled(fill_value=0)
    lats = data['lat'][:]
    lons = data['lon'][:]
    data.close()
    return var, lats, lons


# def read_dem(filename):
#     tif = TIFF.open(filename, mode='r')
#     stack = []
#     for img in list(tif.iter_images()):
#         stack.append(img)
#
#     dem_np = np.array(stack)
#     #     dem_np=np.squeeze(dem_np.transpose(1,2,0))
#
#     dem_np = np.squeeze(dem_np.transpose(1, 2, 0))
#     return dem_np


def add_lat_lon(data, domian=[112.9, 154.25, -43.7425, -9.0], xarray=True):
    "data: is the something you want to add lat and lon, with first demenstion is lat,second dimention is lon,domain is DEM domain "
    new_lon = np.linspace(domian[0], domian[1], data.shape[1])
    new_lat = np.linspace(domian[2], domian[3], data.shape[0])
    if xarray:
        return xr.DataArray(data[:, :, 0], coords=[new_lat, new_lon], dims=["lat", "lon"])
    else:
        return data, new_lat, new_lon


def add_lat_lon_data(data, domain=[112.9, 154.00, -43.7425, -9.0], xarray=True):
    "data: is the something you want to add lat and lon, with first demenstion is lat,second dimention is lon,domain is DEM domain "
    new_lon = np.linspace(domain[0], domain[1], data.shape[1])
    new_lat = np.linspace(domain[2], domain[3], data.shape[0])
    if xarray:
        return xr.DataArray(data, coords=[new_lat, new_lon], dims=["lat", "lon"])
    else:
        return data, new_lat, new_lon


def map_aust_old(data, lat=None, lon=None, domain=[112.9, 154.25, -43.7425, -9.0], xrarray=True):
    '''
    domain=[111.975, 156.275, -44.525, -9.975]
    domain = [111.85, 156.275, -44.35, -9.975]for can be divide by 4
    xarray boolean :the out put data is xrray or not
    '''
    if str(type(data)) == "<class 'xarray.core.dataarray.DataArray'>":
        da = data.data
        lat = data.lat.data
        lon = data.lon.data
    else:
        da = data

    #     if domain==None:
    #         domain = [111.85, 156.275, -44.35, -9.975]
    a = np.logical_and(lon >= domain[0], lon <= domain[1])
    b = np.logical_and(lat >= domain[2], lat <= domain[3])
    da = da[b, :][:, a].copy()
    llons, llats = lon[a], lat[b]  # 将维度按照 x,y 横向竖向
    if str(type(data)) == "<class 'xarray.core.dataarray.DataArray'>" and xrarray:
        return xr.DataArray(da, coords=[llats, llons], dims=["lat", "lon"])
    else:
        return da

    return da, llats, llons

def AWAPcalpercentile(startyear, endyear, p_value):
    filepath = "/scratch/iu60/xs5813/Awap_data_bigger/" 
    pr_value = []
    for file in os.listdir(filepath):
        if startyear <= int(file[:4]) <= endyear:

            dataset = xr.open_dataset(filepath + file)
            dataset = dataset.fillna(0)
            var = dataset.isel(time=0)['precip'].values
            pr_value.append(var)
            dataset.close()
    np_pr_value = np.array(pr_value)
    return np.percentile(np_pr_value, p_value, axis=0)
#give special position
def AWAPcalpercentilePos(startyear, endyear, p_value, lat, lon):
    filepath = "/scratch/iu60/xs5813/Awap_data_bigger/" 
    pr_value = []

    for file in os.listdir(filepath):
        if startyear <= int(file[:4]) <= endyear:
            # 读取数据文件
            dataset = xr.open_dataset(filepath + file)
            dataset = dataset.fillna(0)
            point_data = dataset.sel(lat=lat, lon=lon, method="nearest")
            var = point_data.isel(time=0)['precip'].values
            pr_value.append(var)
            dataset.close()
    # 将收集到的降水数据转换成 NumPy 数组
    np_pr_value = np.array(pr_value)

    # 计算并返回所需的百分位数值
    return np.percentile(np_pr_value, p_value, axis=0)


def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]