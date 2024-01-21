import RRDBNet_arch as arch
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

import argparse


SAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_ORIGINAL/"
TEST_SAVE_PREFIX = "/scratch/iu60/xs5813/TestResults/"
MODEL_PREFIX = SAVE_PREFIX + "DESRGAN/checkpoint/" + "v" 
TESTING_DATA = "/scratch/iu60/xs5813/Processed_data/e*/*.nc"

australia_lon_range = (143,  153.5)
australia_lat_range = (-32, -24)

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

def generate_sample(rain_prob, gamma_shape, gamma_scale):
    # 生成一个样本估计值
    rain_sample = torch.zeros_like(rain_prob)
    rain_mask = torch.bernoulli(rain_prob)  # 根据降雨概率生成一个掩码
    rain_sample[rain_mask.bool()] = torch.distributions.Gamma(gamma_shape, gamma_scale).sample()[rain_mask.bool()]
    return rain_sample

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
    rain_prob, gamma_shape, gamma_scale = model_G(batch_input)

    slice_output = generate_sample(rain_prob, gamma_shape, gamma_scale)

    # Resize and convert back to DataArray
    slice_output = slice_output.cpu().data.numpy()
    slice_output = np.clip(slice_output[0], 0., 1.)
    slice_output = cv2.resize(np.squeeze(slice_output), (366, 381), interpolation=cv2.INTER_CUBIC)
    slice_output = np.clip(slice_output, 0, 1)
    
    return slice_output

def select_data(data, time_value):
    """
    Select precipitation data from the given data array for the given time value over Australia.

    Args:
    - da (xarray.DataArray): Data array to select from.
    - time_value (int): Time value for the data array.

    Returns:
    - xarray.DataArray: Selected data array.
    """
    data = data.sel(time=time_value)['pr']
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
    new_shape = (366,381)  # Target width and height
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

def test_model(model_G_name: str, version: str, year: int) -> None:
    """
        For all ACCESS-S2 data in the given year, run the model and save the results

        Args:
            model_G_name (str): model name
            version (str): version
            year (int): year to test on
    """
    
    # Make directories
    make_directories(model_G_name, version)

    # Model architecture
    model_G = arch.RRDBNetx4x2(1, 3, 64, 23, gc=32).cuda()

    # Load model
    model_path = MODEL_PREFIX + version + "/" + model_G_name + ".pth"
    model_G.load_state_dict(torch.load(model_path).module.state_dict(), strict=True)  

    # Parallel
    if torch.cuda.device_count() > 1:
        model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))
    model_G.eval()

    # Get all files in the given year for all ensemble
    testing_data = glob.glob(TESTING_DATA)
    testing_data = [f for f in testing_data if str(year) in f]

    np.random.shuffle(testing_data)


    # Test
    with torch.no_grad():
        for fn in testing_data:
            # Save path
            en = fn.split("/")[-2]
            startdate = fn.split("/")[-1].split("_")[0].split(".")[0]
            savepath = TEST_SAVE_PREFIX + "DESRGAN/v" + version + "/"  + model_G_name + "/" + en + "/" + startdate + ".nc"
            
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
            for time_value in ds_raw['time'].values:
                print(f"Processing time value: {time_value}")

                # Select time value and Australian region
                da_selected = select_data(ds_preprocessed, time_value)

                # Convert to PyTorch tensor and process
                batch_output = generate_batch(model_G, da_selected)

                # Resize and convert back to DataArray
                da_to_save = resize_data(batch_output, time_value)

                # Add to running list
                ds_total.append(da_to_save)
            
            # Merge all data arrays
            ds_total = xr.concat(ds_total, dim="time")

            # Save the final merged dataset
            ds_total.to_netcdf(savepath)


if __name__ == "__main__":

    #model_G_name = f"model_G_i0000{str(9).zfill(2)}"
    model_G_name = "model_G_i000004_best"
    #version = "TrainingIterationRMSETest"
    version = "TestRefactored"

    test_model(model_G_name, version, 2002)