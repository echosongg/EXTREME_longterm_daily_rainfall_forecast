import pretrain_m_arch as arch
import argparse
import glob
from os import mkdir
from os.path import isdir

import cv2
import numpy as np
import torch
import xarray as xr
from torch.autograd import Variable
from utils2 import generate_sample, generate_3_channels
from datetime import date, timedelta, datetime
import os
# from tqdm import tqdm

import torch.nn as nn
SAVE_PREFIX = "/scratch/iu60/xs5813/EXTREME/"
TEST_SAVE_PREFIX = "/scratch/iu60/xs5813/TestResults/"
MODEL_PREFIX = SAVE_PREFIX + "checkpoint/vversion_0"
TESTING_DATA = "/scratch/iu60/xs5813/Processed_data_bigger/e01/*.nc"
OSAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_YAO"
OMODEL_PREFIX = OSAVE_PREFIX + "/checkpoint/" + "voriginal_DESRGAN" 
version = "TestRefactored"
generate = False

def test(batch_input, generate):
    
    # divide and conquer strategy due to GPU memory limit

    # _, H, W = batch_input.size()
    def resize_output(slice_output):
    # Resize and convert back to DataArray
        print("slice_output shape", slice_output.shape)
        slice_output = slice_output.cpu().data.numpy()
        slice_output = np.clip(slice_output[0], 0., 1.)
        slice_output = cv2.resize(np.squeeze(slice_output), (267, 413), interpolation=cv2.INTER_CUBIC)
        slice_output = np.clip(slice_output, 0, 1)
        return slice_output
    #def resize_output(batch_Out):
    #    batch_Out = batch_Out.cpu().data.numpy()
    #    batch_Out = np.clip(batch_Out, 0., 1.)  # BXCxHxW 16*1*688*880
    #    batch_Out = np.squeeze(batch_Out, axis=1)  # 16*688*880
    #    batch_Out = batch_Out.transpose(1, 2, 0) # 688*880*16

    #    batch_Out = cv2.resize(batch_Out, (267, 413), interpolation=cv2.INTER_CUBIC)
    #    if len(batch_Out.shape) == 2:
    #        batch_Out = batch_Out.reshape(batch_Out.shape[0], batch_Out.shape[1], 1)
    #    # batch_Out = np.transpose(batch_Out, [2, 0, 1])
    #   batch_Out = batch_Out.transpose(2, 0, 1) # 3 * 413 * 267
    #    return batch_Out

    output = model_G(batch_input)
    if generate == True:
        slice_output = generate_sample(output)
        slice_output = resize_output(slice_output)
        print("generate")
        print(slice_output)
        return slice_output
    else: 
        p_pred, alpha_pred, beta_pred = generate_3_channels(output)
    #get slice p, a, b
        p_pred_slice = resize_output(p_pred)
        alpha_pred_slice = resize_output(alpha_pred)
        beta_pred_slice = resize_output(beta_pred)
        return p_pred_slice, alpha_pred_slice, beta_pred_slice
        #暂时先只要p, 后面可以画a, b扩大后的图像

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
    for en in ["e01","e02","e03","e04","e05","e06","e07","e08","e09"]:
        if not isdir(TEST_SAVE_PREFIX + "DESRGAN" + "/" + "v" + version + "/" + model_name.split(".")[0] + "/" + en):
            mkdir(TEST_SAVE_PREFIX + "DESRGAN" + "/" + "v" + version + "/" + model_name.split(".")[0] + "/" + en)

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('-n', type=int, default=2,
                    help='Divisor, make larger when GPU memory shortage')
args = parser.parse_args()
model_name = 'vVersion1.6'
years = ['2006', '2007', '2008', '2009', '2010','2011', '2018']  # 定义需要处理的年份列表
version = "TestRefactored"
# Model
omodel_G_name = "model_G_i000005_20240401-025017"
# Load model
omodel_path = OMODEL_PREFIX +  "/" + omodel_G_name + ".pth"
for year in years:  # 外层循环遍历每一个年份
    for model in ['model_G_i000005_20240403-035316_with_huber']:
        make_directories(model, version)
        model_path = MODEL_PREFIX + "/" + model + ".pth"
        model_G = arch.ModifiedRRDBNet(omodel_path, 1, 3, 64, 23, gc=32).cuda()
        model_G.load_state_dict(torch.load(model_path).module.state_dict(), strict=True)  
        if torch.cuda.device_count() > 1:
            print("!!!Let's use " + str(torch.cuda.device_count()) + "GPUs!")
            model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))
        model_G.eval()

        # Test
        with torch.no_grad():
            # Input nc file
            # ['e05', 'e06', 'e08']
            exx = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09']
            path = '/scratch/iu60/xs5813/Processed_data_bigger/e*/*.nc'
            allfiles = glob.glob(path)

            for en in exx:
                files = []

                # for i in allfiles:
                #     if "da_pr_2009" in i or "da_pr_2011" in i:
                #         if en in i:
                #             files.append(i)
                for i in allfiles:
                    if ("da_pr_" + year) in i:
                        if en in i:
                            files.append(i)

                files.sort()
                for fn in files:
                    ds_raw = xr.open_dataset(fn) * 86400
                    ds_raw = ds_raw.fillna(0)
                    da_selected = ds_raw.isel(time=0)["pr"]
                    #caculate_time
                    startdate = str(ds_raw['time'].values[0])[:10]
                    # To 0-1

                    lon = ds_raw["lon"].values
                    lat = ds_raw["lat"].values
                    a = np.logical_and(lon >= 140.6, lon <= 153.9)
                    b = np.logical_and(lat >= -39.2, lat <= -18.6)
                    da_selected_au = da_selected[b, :][:, a].copy()
                    print("odata",da_selected_au)
                    n = 1
                    # lat691 lon886
                    size = (int(267), int(413))
                    # size = (int(632), int(728))
                    new_lon = np.linspace(
                        da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
                    new_lon = np.float32(new_lon)
                    new_lat = np.linspace(
                        da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
                    new_lat = np.float32(new_lat)
                    da_selected_pr = da_selected_au.values
                    da_selected_pr = np.clip(da_selected_pr, 0, 1000)
                    da_selected_pr = np.log1p(da_selected_pr) / 7
                    da_selected_pr = cv2.resize(
                        da_selected_pr, (33, 51), interpolation=cv2.INTER_CUBIC)

                    da_selected_pr = np.asarray(da_selected_pr).astype(np.float32)
                    da_selected_pr = da_selected_pr[np.newaxis, np.newaxis, ...]
                    batch_input = Variable(torch.from_numpy(da_selected_pr)).cuda()
                    # Save to file
                    i = ds_raw['time'].values[0]
                    if generate == True:
                        batch_output = test(batch_input, generate)
                        da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),
                                                coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
                        ds_total = xr.concat([da_interp], "time")
                    else:
                        #p_pred, batch_output = test(batch_input, generate)
                        #da_interp_p = xr.DataArray(p_pred, dims=("lat", "lon"),
                                        #coords={"lat": new_lat, "lon": new_lon, "time": i}, name='p')
                        #ds_p_total = xr.concat([da_interp_p], "time")
                        p_pred, alpha_pred, beta_pred = test(batch_input, generate)
                        da_interp_p = xr.DataArray(p_pred, dims=("lat", "lon"),
                                        coords={"lat": new_lat, "lon": new_lon, "time": i}, name='p')
                        da_interp_alpha = xr.DataArray(alpha_pred, dims=("lat", "lon"),
                                        coords={"lat": new_lat, "lon": new_lon, "time": i}, name='alpha')
                        da_interp_beta = xr.DataArray(beta_pred, dims=("lat", "lon"),
                                        coords={"lat": new_lat, "lon": new_lon, "time": i}, name='beta')
                        ds_p_total = xr.concat([da_interp_p], "time")
                        ds_alpha_total = xr.concat([da_interp_alpha], "time")
                        ds_beta_total = xr.concat([da_interp_beta], "time")

                    for i in ds_raw['time'].values[0:50]:
                        ds_selected_domained = ds_raw.sel(time=i)['pr'].values
                        da_selected_au = ds_selected_domained[b, :][:, a].copy()

                        da_selected_pr = da_selected_au
                        da_selected_pr = np.clip(da_selected_pr, 0, 1000)
                        da_selected_pr = np.log1p(da_selected_pr) / 7

                        da_selected_pr = cv2.resize(
                            da_selected_pr, (33, 51), interpolation=cv2.INTER_CUBIC)

                        da_selected_pr = np.asarray(da_selected_pr).astype(np.float32)
                        da_selected_pr = da_selected_pr[np.newaxis, np.newaxis, ...]
                        batch_input = Variable(torch.from_numpy(da_selected_pr)).cuda()
                        if generate == True:
                            batch_output = test(batch_input, generate)
                            da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),
                                            coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
                            expanded_da = xr.concat([da_interp], "time")
                            ds_total = xr.merge([ds_total, expanded_da])
                        else:
                            p_pred, alpha_pred, beta_pred = test(batch_input, generate)
                            da_interp_p = xr.DataArray(p_pred, dims=("lat", "lon"),
                                            coords={"lat": new_lat, "lon": new_lon, "time": i}, name='p')
                            expanded_p = xr.concat([da_interp_p], "time")
                            da_interp_alpha = xr.DataArray(alpha_pred, dims=("lat", "lon"),
                                            coords={"lat": new_lat, "lon": new_lon, "time": i}, name='alpha')
                            expanded_alpha = xr.concat([da_interp_alpha], "time")
                            da_interp_beta = xr.DataArray(beta_pred, dims=("lat", "lon"),
                                            coords={"lat": new_lat, "lon": new_lon, "time": i}, name='beta')
                            expanded_beta = xr.concat([da_interp_beta], "time")
                            ds_p_total = xr.merge([ds_p_total, expanded_p])
                            ds_alpha_total = xr.merge([ds_alpha_total, expanded_alpha])
                            ds_beta_total = xr.merge([ds_beta_total, expanded_beta])
                    savepath_pr = TEST_SAVE_PREFIX  + "DESRGAN/v" + version + "/"  + model + "/" + en + "/"+  "pr" + "/" + year + "/" + startdate + ".nc"
                    savepath_p = TEST_SAVE_PREFIX  + "DESRGAN/v" + version + "/"  + model + "/" + en + "/"+  "p" +"/" + year + "/" + startdate + ".nc"
                    savepath_alpha = TEST_SAVE_PREFIX  + "DESRGAN/v" + version + "/"  + model + "/" + en + "/"+  "alpha" +"/" + year + "/" + startdate + ".nc"
                    savepath_beta = TEST_SAVE_PREFIX  + "DESRGAN/v" + version + "/"  + model + "/" + en + "/"+  "beta" +"/" + year + "/" + startdate + ".nc"
                    def ensure_directory_exists(path):
                        directory = os.path.dirname(path)
                        if not os.path.exists(directory):
                            os.makedirs(directory, exist_ok=True)

                    # 确保所有需要的路径都存在
                    ensure_directory_exists(savepath_pr)
                    ensure_directory_exists(savepath_p)
                    ensure_directory_exists(savepath_alpha)
                    ensure_directory_exists(savepath_beta)
                    if generate == True:
                        ds_total.to_netcdf(savepath_pr)

                    if generate == False:
                        ds_p_total.to_netcdf(savepath_p)
                        ds_alpha_total.to_netcdf(savepath_alpha)
                        ds_beta_total.to_netcdf(savepath_beta)
