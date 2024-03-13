import RRDBNet_arch2 as arch
import argparse
import glob
from os import mkdir
from os.path import isdir

import cv2
import numpy as np
import torch
import xarray as xr
from torch.autograd import Variable
import os
import torch.nn as nn
from datetime import date, timedelta, datetime
import argparse


SAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_ORIGINAL/"
TEST_SAVE_PREFIX = "/scratch/iu60/xs5813/TestResults/"
MODEL_PREFIX = SAVE_PREFIX + "DESRGAN/checkpoint/" + "v" 
TESTING_DATA = "/scratch/iu60/xs5813/Processed_data/e01/*.nc"

#australia_lon_range = (142.000,  152.3)
#australia_lat_range = (-31.95, -23.4)
australia_lon_range = (140.6, 153.9)
australia_lat_range = (-39.2, -18.6)
scale = 1.5

def make_directories(model_name: str, version:str) -> None:
    """
        Make directories for saving test results

        Args:
            model_name (str): model name
    """

    if not isdir(TEST_SAVE_PREFIX + "DESRGAN"):
        mkdir(TEST_SAVE_PREFIX + "DESRGAN")
    if not isdir(TEST_SAVE_PREFIX + "DESRGAN"+ "/" + "v" + version):
        mkdir(TEST_SAVE_PREFIX + "DESRGAN"+ "/" + "v" + version)
    if not isdir(TEST_SAVE_PREFIX + "DESRGAN"+ "/" + "v" + version + "/" + model_name.split(".")[0]):
        mkdir(TEST_SAVE_PREFIX + "DESRGAN"+ "/" + "v" + version + "/" + model_name.split(".")[0])
    #for en in ["e01", "e02", "e03"]:
    for en in ["e01"]:
        if not isdir(TEST_SAVE_PREFIX + "DESRGAN" + "/" + "v" + version + "/" + model_name.split(".")[0] + "/" + en):
            mkdir(TEST_SAVE_PREFIX + "DESRGAN" + "/" + "v" + version + "/" + model_name.split(".")[0] + "/" + en)



def generate_batch(model_G, da_selected):
    """
        Run the model on the given batch input

        Args:
            da_selected (xarray.DataArray): batch input
            model_G (nn.Module): model to use for processing
        
        Returns:
            slice_output (np.ndarray): batch output
    """

    # Convert to PyTorch tensor and process
    da_tensor = np.asarray(da_selected).astype(np.float32)
    da_tensor = da_tensor[np.newaxis, np.newaxis, ...]  # Add batch and channel dimensions
    batch_input = Variable(torch.from_numpy(da_tensor)).cuda()

    # Run model
    #slice_output = model_G(batch_input)
    slice_output = model_G(batch_input)
    # Resize and convert back to DataArray
    slice_output = slice_output.cpu().data.numpy()
    slice_output = np.clip(slice_output[0], 0., 1.)
    #207,172
    slice_output = cv2.resize(np.squeeze(slice_output), (267, 413), interpolation=cv2.INTER_CUBIC)
    slice_output = np.clip(slice_output, 0, 1)
    
    return slice_output

def select_data(data, leadingtime):
    """
    Select precipitation data from the given data array for the given time value over Australia.

    Args:
    - da (xarray.DataArray): Data array to select from.
    - time_value (int): Time value for the data array.

    Returns:
    - xarray.DataArray: Selected data array.
    """
    data = data.isel(time=leadingtime)['pr']
    lon_mask = np.logical_and(data.lon >= australia_lon_range[0], data.lon <= australia_lon_range[1])
    lat_mask = np.logical_and(data.lat >= australia_lat_range[0], data.lat <= australia_lat_range[1])
    return data.sel(lon=lon_mask, lat=lat_mask)

def resize_data(data, time_value):
    """
    Resize data array to new shape.

    Args:
    - data (xarray.DataArray): Data array to resize.
    - time_value (numpy.datetime64): Time value for the data array.

    Returns:
    - xarray.DataArray: Resized data array.
    """

    # Resize data to target shape
    #new_shape = (215,161)  # Target width and height
    #new_shape = (207,172)
    new_shape = (267, 413)
    resized_values = np.asarray(data).astype(np.float32)
    resized_values = cv2.resize(resized_values, new_shape, interpolation=cv2.INTER_CUBIC)
    if not isinstance(time_value, np.datetime64):
        # 尝试将其转换为 np.datetime64
        time_value = np.datetime64(time_value)

    # Create new coordinates for the resized data array
    coords = {
        "lat": np.linspace(australia_lat_range[0], australia_lat_range[1], new_shape[1]),
        "lon": np.linspace(australia_lon_range[0], australia_lon_range[1], new_shape[0]),
        "time": np.datetime_as_string(time_value, unit='D') # YYYY-MM-DD
    }

    # Return resized data array
    return xr.DataArray(resized_values, dims=("lat", "lon"), coords=coords, name='pr')

def test_model(model_G_name: str, version: str, year: int, month: int, leadingtime) -> None:
    """
        For all ACCESS-S2 data in the given year, run the model and save the results

        Args:
            model_G_name (str): model name
            version (str): version
            year (int): year to test on
            month(int): month to test on
    """
    
    # Make directories
    make_directories(model_G_name, version)

    # Model architecture
    model_G = arch.RRDBNetx4x2(1, 1, 64, 23, gc=32).cuda()

    # Load model
    model_path = MODEL_PREFIX + version + "/" + model_G_name + ".pth"
    model_G.load_state_dict(torch.load(model_path).module.state_dict(), strict=True)  

    # Parallel
    if torch.cuda.device_count() > 1:
        model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))
    model_G.eval()

    # Get all files in the given year for all ensemble
    testing_data = glob.glob(TESTING_DATA)
    testing_data = [f for f in testing_data if f"{year}-{str(month).zfill(2)}" in f]


    np.random.shuffle(testing_data)


    # Test
    print("start ...")
    with torch.no_grad():
        for fn in testing_data:
            print("bb")
            # Save path
            en = fn.split("/")[-2]
            startdate = fn.split("/")[-1].split("_")[0].split(".")[0]
            startdate = datetime.strptime(startdate, "%Y-%m-%d")
            # 加一天
            enddate = startdate + timedelta(days=leadingtime)
            enddate = enddate.strftime("%Y-%m-%d")
            savepath = TEST_SAVE_PREFIX + "DESRGAN/v" + version + "/"  + model_G_name + "/" + en + "/"+ "/" + "pr" + "/" + enddate + ".nc"
            
            save_dir = os.path.dirname(savepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # Skip if already exists
            if os.path.exists(savepath):
                print(f"Already exists: {savepath}")
                continue
            print("fn")
            print(fn)
            #/scratch/iu60/xs5813/Processed_data/e05/2002-01-31.nc
            #save path 
            #/scratch/iu60/xs5813/TestResults/DESRGAN/vTestRefactored/model_G_i000005/e05/2002-01-31.nc'
            # Get raw data and preprocess
            ds_raw = xr.open_dataset(fn)
            ds_preprocessed = ds_raw.map(np.log1p) / 7
            
            ds_total = []
                # Select time value and Australian region
            da_selected = select_data(ds_preprocessed, leadingtime)

                # Convert to PyTorch tensor and process
            batch_output = generate_batch(model_G, da_selected)

                # Resize and convert back to DataArray
            da_to_save = resize_data(batch_output, enddate)
            
            # Merge all data arrays
            da_to_save = xr.concat(da_to_save, dim="time")

            # Save the final merged dataset
            da_to_save.to_netcdf(savepath)


if __name__ == "__main__":

    #model_G_name = f"model_G_i0000{str(9).zfill(2)}"
    model_G_name = "model_G_i000010_20240307-155833"
    print("original model name summer 1-6", model_G_name)
    #version = "TrainingIterationRMSETest"
    version = "TestRefactored"

    test_model(model_G_name, version, 2002, 12, 1)