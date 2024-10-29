import os
from os.path import isdir

import data_processing_tool_qm as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
import torch, torchvision
import numpy as np
import random

from torch.utils.data import Dataset, random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
from PIL import Image

import torch
import matplotlib as plt
import argparse
import sys
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import platform
from datetime import timedelta, date, datetime
import numpy as np
import os
import time
# from sklearn.metrics import mean_absolute_error
import statistics
import properscoring as ps


# ===========================================================
# Training settings
# ===========================================================
def calculate_pit_values_plus(ensemble_forecasts, observations, epsilon=1e-6):
    """
    Calculate PIT (Probability Integral Transform) values for 3D ensemble forecasts.

    Parameters:
    ensemble_forecasts (array-like): A 3D array where dimensions represent:
                                     (Horizontal axis, Vertical axis, Ensemble member)
    observations (array-like): A 2D array of corresponding observed values with dimensions:
                               (Horizontal axis, Vertical axis)
    epsilon (float): A tiny value to adjust PIT away from extremes (default: 1e-6)

    Returns:
    array: A 2D array of PIT values with the same shape as observations.
    """
    # Ensure inputs are numpy arrays
    ensemble_forecasts = np.array(ensemble_forecasts)
    observations = np.array(observations)
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))

    # Check input shapes
    if ensemble_forecasts.ndim != 3 or observations.ndim != 2:
        raise ValueError(
            "Input dimensions are incorrect. ensemble_forecasts should be 3D and observations should be 2D.")

    if ensemble_forecasts.shape[:2] != observations.shape:
        raise ValueError("Horizontal and vertical dimensions of ensemble_forecasts and observations must match.")

    # Count the total number of ensemble members
    n_ensemble = ensemble_forecasts.shape[-1]+1

    # Sort each ensemble forecast
    sorted_forecasts = np.sort(ensemble_forecasts, axis=-1)

    # Calculate left and right ranks
    left_ranks = np.zeros_like(observations, dtype=int)
    right_ranks = np.zeros_like(observations, dtype=int)

    for i in range(observations.shape[0]):
        for j in range(observations.shape[1]):
            left_ranks[i, j] = np.searchsorted(sorted_forecasts[i, j], observations[i, j], side='left')
            right_ranks[i, j] = np.searchsorted(sorted_forecasts[i, j], observations[i, j], side='right')

    # Generate random values for cases where left_rank != right_rank
    random_values = np.random.random(observations.shape)

    # Calculate PIT values
    pit_values = np.where(
        left_ranks != right_ranks,
        (left_ranks + random_values * (right_ranks - left_ranks)) / n_ensemble,
        left_ranks / n_ensemble
    )

    # Apply the tiny perturbation to avoid extreme values of 0 and 1
    pit_values = np.clip(pit_values, epsilon, 1 - epsilon)

    return pit_values

def calculate_pit_values(ensemble_forecasts, observations, epsilon=1e-6):
    """
    Calculate PIT (Probability Integral Transform) values for 3D ensemble forecasts.

    Parameters:
    ensemble_forecasts (array-like): A 3D array where dimensions represent:
                                     (Horizontal axis, Vertical axis, Ensemble member)
    observations (array-like): A 2D array of corresponding observed values with dimensions:
                               (Horizontal axis, Vertical axis)
    epsilon (float): A tiny value to adjust PIT away from extremes (default: 1e-6)

    Returns:
    array: A 2D array of PIT values with the same shape as observations.
    """
    # Ensure inputs are numpy arrays
    ensemble_forecasts = np.array(ensemble_forecasts)
    observations = np.array(observations)
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))

    # Check input shapes
    if ensemble_forecasts.ndim != 3 or observations.ndim != 2:
        raise ValueError(
            "Input dimensions are incorrect. ensemble_forecasts should be 3D and observations should be 2D.")

    if ensemble_forecasts.shape[:2] != observations.shape:
        raise ValueError("Horizontal and vertical dimensions of ensemble_forecasts and observations must match.")

    # Count the total number of ensemble members
    n_ensemble = ensemble_forecasts.shape[-1]

    # Sort each ensemble forecast
    sorted_forecasts = np.sort(ensemble_forecasts, axis=-1)

    # Calculate left and right ranks
    left_ranks = np.zeros_like(observations, dtype=int)
    right_ranks = np.zeros_like(observations, dtype=int)

    for i in range(observations.shape[0]):
        for j in range(observations.shape[1]):
            left_ranks[i, j] = np.searchsorted(sorted_forecasts[i, j], observations[i, j], side='left')
            right_ranks[i, j] = np.searchsorted(sorted_forecasts[i, j], observations[i, j], side='right')

    # Generate random values for cases where left_rank != right_rank
    random_values = np.random.random(observations.shape)

    # Calculate PIT values
    pit_values = np.where(
        left_ranks != right_ranks,
        (left_ranks + random_values * (right_ranks - left_ranks)) / n_ensemble,
        left_ranks / n_ensemble
    )

    # Apply the tiny perturbation to avoid extreme values of 0 and 1
    pit_values = np.clip(pit_values, epsilon, 1 - epsilon)

    return pit_values

def calculate_alpha_index(pit_values):
    """
    Optimized calculation of the alpha index for forecast reliability based on 3D PIT values.

    Parameters:
    pit_values (array-like): A 3D array of PIT (Probability Integral Transform) values,
                             where the first dimension represents starting times and the
                             second and third dimensions are spatial (x, y).

    Returns:
    np.ndarray: A 2D array of alpha index values where each element corresponds
                to a spatial point (x, y) in the grid.
    """
    # Get the shape of the PIT values array
    date_size, x_size, y_size = pit_values.shape

    # Sort the PIT values along the ensemble dimension (axis 0)
    pit_sorted = np.sort(pit_values, axis=0)

    # Calculate expected uniform distribution values for the ensemble members (broadcasted)
    expected_uniform = np.linspace(1 / (date_size + 1), date_size / (date_size + 1), date_size)

    # Calculate the absolute differences between sorted PIT values and the expected uniform distribution
    absolute_differences = np.abs(pit_sorted - expected_uniform[:, None, None])

    # Sum the absolute differences along the ensemble member dimension (axis 0)
    sum_absolute_differences = np.sum(absolute_differences, axis=0)

    # Calculate the alpha index for each spatial point
    alpha_values = 1 - (2 / date_size) * sum_absolute_differences

    return alpha_values

class ACCESS_AWAP_cali(Dataset):
    '''

2.using my net to train one channel to one channel.

    '''

    def __init__(self, start_date=date(1990, 1, 1), end_date=date(1990, 12, 31), regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, args=None):
        #         print("=> BARRA_R & ACCESS_S1 loading")
        #         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_AWAP_dir = args.file_AWAP_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args = args

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date

        self.regin = regin
        self.leading_time_we_use = args.leading_time_we_use

        self.ensemble_access = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09']
        self.ensemble = []
        for i in range(len(self.ensemble_access)):
            self.ensemble.append(self.ensemble_access[i])

        self.dates = self.date_range(start_date, end_date)
        print("rootdir:", self.file_ACCESS_dir)
        self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)
        print("inluding files:", self.filename_list)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir)
            print("no file or no permission")

        en, cali_date, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        data_awap = dpt.read_awap_data_fc_get_lat_lon(self.file_AWAP_dir, date_for_AWAP)
        self.lat_awap = data_awap[1]
        self.lon_awap = data_awap[2]

        # data_cali = dpt.read_access_data_calibrataion_get_lat_lon(self.file_ACCESS_dir, en, cali_date, 0)
        # self.lat_cali = data_cali[1]
        # self.lon_cali = data_cali[2]

    #         self.shape=(316, 376)

    #         self.data_dem=dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif")

    def __len__(self):
        return len(self.filename_list)

    def date_range(self, start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    def get_filename_with_no_time_order(self, rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:] == ".nc":
                    _files.append(path)
        return _files

    def get_filename_with_time_order(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use, self.leading_time_we_use + 1):
                print("leading time we use", i)
                for en in self.ensemble:
                    access_path = rootdir + "e09" + "/" + "daq5_pr_" + date.strftime("%Y%m%d") + "_" + "e09" + ".nc"
                    print("access file", access_path)
                    if os.path.exists(access_path):
                        print("The file exisits in e09")
                        if date == self.end_date and i == 1:
                            break
                        path = []
                        path.append(en)
                        awap_date = date + timedelta(i)
                        path.append(date)
                        path.append(awap_date)
                        path.append(i)
                        _files.append(path)

        # 最后去掉第一行，然后shuffle
        return _files

    def mapping(self, X, min_val=0., max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        # 将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b - a) / (Xmax - Xmin) * (X - Xmin)
        return Y

    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t = time.time()

        # read_data filemame[idx]
        en, access_date, awap_date, time_leading = self.filename_list[idx]

        lr = dpt.read_access_data_calibrataion(self.file_ACCESS_dir, en, access_date, time_leading, "pr")
        label, AWAP_date = dpt.read_awap_data_fc(self.file_AWAP_dir, awap_date)

        return np.array(lr), np.array([1]), self.hr_transform(Image.fromarray(label)), torch.tensor(
            int(en[1:])), torch.tensor(int(access_date.strftime("%Y%m%d"))), torch.tensor(time_leading)


def write_log(log, args):
    print(log)
    # 确保目录存在，如果不存在则创建所有必要的父目录
    log_dir = f"./save/{args.train_name}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 安全地打开文件并写入日志
    log_file_path = os.path.join(log_dir, 'train.txt')
    with open(log_file_path, 'a') as my_log_file:
        my_log_file.write(log + '\n')


def main(year, days):
    Brier_startyear = 1976
    Brier_endyear = 2005
    #percentile_999 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99.9)
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true', help='cpu only?')

    # hyper-parameters
    parser.add_argument('--train_name', type=str, default="cali_crps", help='training name')

    parser.add_argument('--batch_size', type=int, default=9, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

    # model configuration
    parser.add_argument('--upscale_factor', '-uf', type=int, default=4, help="super resolution upscale factor")
    parser.add_argument('--model', '-m', type=str, default='vdsr', help='choose which model is going to use')

    # data
    parser.add_argument('--pr', type=bool, default=True, help='add-on pr?')

    parser.add_argument('--train_start_time', type=type(datetime(1990, 1, 25)), default=datetime(1990, 1, 2), help='r?')
    parser.add_argument('--train_end_time', type=type(datetime(1990, 1, 25)), default=datetime(1990, 2, 9), help='?')
    parser.add_argument('--test_start_time', type=type(datetime(2017, 1, 1)), default=datetime(2007, 1, 1), help='a?')
    parser.add_argument('--test_end_time', type=type(datetime(2007, 12, 31)), default=datetime(2007, 12, 31), help='')

    parser.add_argument('--dem', action='store_true', help='add-on dem?')
    parser.add_argument('--psl', action='store_true', help='add-on psl?')
    parser.add_argument('--zg', action='store_true', help='add-on zg?')
    parser.add_argument('--tasmax', action='store_true', help='add-on tasmax?')
    parser.add_argument('--tasmin', action='store_true', help='add-on tasmin?')
    parser.add_argument('--leading_time_we_use', type=int, default=1
                        , help='add-on tasmin?')
    parser.add_argument('--ensemble', type=int, default=9, help='total ensambles is 9')
    parser.add_argument('--channels', type=float, default=0, help='channel of data_input must')
    # [111.85, 155.875, -44.35, -9.975]
    parser.add_argument('--domain', type=list, default=[111.975, 156.275, -44.525, -9.975], help='dataset directory')

    parser.add_argument('--file_ACCESS_dir', type=str,
                        default="/scratch/iu60/xs5813/QM_cropped_data/", help='dataset directory')
    parser.add_argument('--file_AWAP_dir', type=str, default="/scratch/iu60/xs5813/Awap_data_bigger/",
                        help='dataset directory')
    # parser.add_argument('--file_DEM_dir', type=str, default="../DEM/",help='dataset directory')
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half', 'double'),
                        help='FP precision for test (single | half)')

    args = parser.parse_args()

    # def main():

    #     init_date=date(1970, 1, 1)
    #     start_date=date(1990, 1, 2)
    #     end_date=date(2011,12,25)
    sys = platform.system()
    args.dem = False
    args.train_name = "pr_dem"
    args.channels = 0
    if args.pr:
        args.channels += 1
    if args.zg:
        args.channels += 1
    if args.psl:
        args.channels += 1
    if args.tasmax:
        args.channels += 1
    if args.tasmin:
        args.channels += 1
    if args.dem:
        args.channels += 1
    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s" % args.train_name)
    print("  ------------------------------")
    print("  num of channels | %5d" % args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d" % args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d" % args.batch_size)
    print("  ------------------------------")
    print("  using cpu only | %5d" % args.cpu)

    lr_transforms = transforms.Compose([
        #    transforms.Resize((691, 886)),
        #     transforms.RandomResizedCrop(IMG_SIZE),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(30),
        transforms.ToTensor()
        #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    hr_transforms = transforms.Compose([
        #         transforms.Resize((316, 376)),
        #     transforms.RandomResizedCrop(IMG_SIZE),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(30),
        transforms.ToTensor()
        #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    args.test_start_time = datetime(year, 1, 1)
    args.test_end_time = datetime(year, 12, 31)

    write_log("start", args)

    for lead in range(0, days):
        args.leading_time_we_use = lead

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms,
                                    hr_transform=hr_transforms, shuffle=False, args=args)

        test_data = DataLoader(data_set,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=args.n_threads, drop_last=False)
        pit = []

        for batch, (pr, dem, hr, en, data_time, idx) in enumerate(test_data):

            with torch.set_grad_enabled(False):

                #                 sr = net(pr)
                sr_np = pr.cpu().numpy()
                hr_np = hr.cpu().numpy()

                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(
                        sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b = np.squeeze(hr_np[i * args.ensemble])

                    QM_pit = calculate_pit_values(a, b)
                    pit.append(QM_pit)

        # if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/bias/QM/" + str(year)):
        #     os.mkdir("/scratch/iu60/rw6151/new_crps/save/bias/QM/" + str(year))
        # np.save("/scratch/iu60/rw6151/new_crps/save/bias/QM/" + str(year) + "/lead_time" + str(lead) + '_whole',
        #         np.mean(mean_bias_model, axis=0))

        # if not os.path.exists("/scratch/iu60/rw6151/new_crps/save/bias_median/QM/" + str(year)):
        #     os.mkdir("/scratch/iu60/rw6151/new_crps/save/bias_median/QM/" + str(year))
        # np.save("/scratch/iu60/rw6151/new_crps/save/bias_median/QM/" + str(year) + "/lead_time" + str(lead) + '_whole',
        #        np.mean(mean_bias_median_model, axis=0))
# 设置基础路径
        base_path = "/scratch/iu60/xs5813/qm/new_crps/save"

        # 定义需要创建的文件夹列表
        folders = ["alpha_no_plus/QM"]
        pit = np.stack(pit, axis=0)
        # 循环创建所需的文件夹路径
        for folder in folders:
            folder_path = os.path.join(base_path, folder, str(year))
            os.makedirs(folder_path, exist_ok=True)  # 创建文件夹，如果已存在则忽略
            data_to_save = {
                "alpha_no_plus/QM": calculate_alpha_index(pit)
            }
            #print(f"Average of {folder}: {np.mean(data_to_save[folder])}")
            np.save(os.path.join(folder_path, f"lead_time{lead}_whole"), data_to_save[folder])


if __name__ == '__main__':
    main(year=2018, days=42)
    main(year=2006, days=42)
    main(year=2007, days=42)
    # print('2018done')
    # main(year=2009, days=42)
    # print('2009done')
    