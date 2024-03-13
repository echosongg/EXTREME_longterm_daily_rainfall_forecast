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
# from tqdm import tqdm
from utils import generate_sample, generate_3_channels

import torch.nn as nn
#这个代码存储了1-217 leadingtime
SAVE_PREFIX = "/scratch/iu60/xs5813/EXTREME_MODEL_VERSION/"
TEST_SAVE_PREFIX = "/scratch/iu60/xs5813/TestResults/"
MODEL_PREFIX = SAVE_PREFIX + "version_1/checkpoint/" + "v" 
TESTING_DATA = "/scratch/iu60/xs5813/Processed_data/e01/*.nc"
OSAVE_PREFIX = "/scratch/iu60/xs5813/DESRGAN_ORIGINAL/"
OMODEL_PREFIX = OSAVE_PREFIX + "DESRGAN/checkpoint/" + "vTestRefactored" 


def test(batch_input):
    # divide and conquer strategy due to GPU memory limit

    # _, H, W = batch_input.size()

    output = model_G(batch_input)
    slice_output = generate_sample(output)

    slice_output = slice_output.cpu().data.numpy()

    slice_output = np.clip(slice_output[0], 0., 1.)

    slice_output = cv2.resize(np.squeeze(slice_output), (207, 172),
                              interpolation=cv2.INTER_CUBIC)

    slice_output = np.clip(slice_output, 0, 1)

    return slice_output


parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('-n', type=int, default=2,
                    help='Divisor, make larger when GPU memory shortage')
args = parser.parse_args()
model_name = 'EXTREME'
year = 2006
# Model
for model in ['model_G_i000012_best_20240228-202147']:
    omodel_G_name = "model_G_i000004_best_20240219-114819"
    # Load model
    omodel_path = OMODEL_PREFIX +  "/" + omodel_G_name + ".pth"

    # Model architecture
    model_G = arch.ModifiedRRDBNet(omodel_path, 1, 3, 64, 23, gc=32).cuda()

    # Load model
    model_path = MODEL_PREFIX + "TestRefactored" + "/" + model + ".pth"
    model_G.load_state_dict(torch.load(model_path).module.state_dict(), strict=True)  

    # Parallel
    if torch.cuda.device_count() > 1:
        model_G = nn.DataParallel(model_G, range(torch.cuda.device_count()))
    model_G.eval()

    # Test
    with torch.no_grad():
        # Input nc file
        # ['e05', 'e06', 'e08']
        exx = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09']
        allfiles = glob.glob(TESTING_DATA)
        files = [f for f in allfiles if str(year) in f]
        np.random.shuffle(files)
        for en in exx:
            files = []
            for i in allfiles:
                if en in i:
                    files.append(i)
            files.sort()
            for fn in files:
                ds_raw = xr.open_dataset(fn)
                ds_raw = ds_raw.fillna(0)
                da_selected = ds_raw.isel(time=0)["pr"]

                startdate = str(ds_raw['time'].values[0])[:10]
                # To 0-1

                lon = ds_raw["lon"].values
                lat = ds_raw["lat"].values
                a = np.logical_and(lon >= 142, lon <= 152.3)
                b = np.logical_and(lat >= -31.95, lat <= -23.4)
                da_selected_au = da_selected[b, :][:, a].copy()
                print(da_selected_au)
                n = 1
                # lat691 lon886
                size = (int(207), int(172))
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
                #da_selected_pr = cv2.resize(
                #    da_selected_pr, (33, 51), interpolation=cv2.INTER_CUBIC)

                da_selected_pr = np.asarray(da_selected_pr).astype(np.float32)
                da_selected_pr = da_selected_pr[np.newaxis, np.newaxis, ...]
                batch_input = Variable(torch.from_numpy(da_selected_pr)).cuda()
                print(batch_input)
                # Save to file
                i = ds_raw['time'].values[0]
                batch_output = test(batch_input)
                da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),
                                         coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
                ds_total = xr.concat([da_interp], "time")

                for i in ds_raw['time'].values[1:217]:
                    ds_selected_domained = ds_raw.sel(time=i)['pr'].values
                    da_selected_au = ds_selected_domained[b, :][:, a].copy()

                    da_selected_pr = da_selected_au
                    da_selected_pr = np.clip(da_selected_pr, 0, 1000)
                    da_selected_pr = np.log1p(da_selected_pr) / 7

                    #da_selected_pr = cv2.resize(
                    #    da_selected_pr, (33, 51), interpolation=cv2.INTER_CUBIC)

                    da_selected_pr = np.asarray(da_selected_pr).astype(np.float32)
                    da_selected_pr = da_selected_pr[np.newaxis, np.newaxis, ...]
                    batch_input = Variable(torch.from_numpy(da_selected_pr)).cuda()

                    batch_output = test(batch_input)
                    da_interp = xr.DataArray(np.expm1(batch_output * 7), dims=("lat", "lon"),
                                             coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
                    expanded_da = xr.concat([da_interp], "time")
                    ds_total = xr.merge([ds_total, expanded_da])
                base_dir = "/scratch/iu60/xs5813/Test_data"

                # Create the model name directory if it doesn't exist
                model_dir = f'{base_dir}/{model_name}'
                if not isdir(model_dir):
                    mkdir(model_dir)

                # Create the year directory within the model name directory if it doesn't exist
                year_dir = f'{model_dir}/{year}'
                if not isdir(year_dir):
                    mkdir(year_dir)

                # Create a directory for the model (limited to the first 15 characters) within the year directory if it doesn't exist
                model_subdir = f'{year_dir}/{model[:15]}'
                if not isdir(model_subdir):
                    mkdir(model_subdir)

                # Finally, create a directory for the 'en' variable within the model directory, if it doesn't exist
                final_dir = f'{model_subdir}/{en}'
                if not isdir(final_dir):
                    mkdir(final_dir)

                # Construct the save path using the new directories
                savepath = f"{final_dir}/{startdate}_{en}.nc"

                # Save the dataset to the constructed path
                ds_total.to_netcdf(savepath)
