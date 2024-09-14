import os
import random
from datetime import date, timedelta

import torch
import torch.nn as nn

import time
import numpy as np
import xarray as xr
import cv2
from torch.utils.data import Dataset


### COMMON FUNCTIONS ###


def dumplicatearray(data, num_repeats):
    return np.dstack([data] * num_repeats)


def PSNR(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(1000. / rmse)


def RMSE(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return rmse

def MAE(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    mae = np.mean(np.abs(diff))

    return mae

def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear

    if reduce:
        return torch.mean(losses)
    else:
        return losses


def im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]


class ACCESS_AWAP_GAN(Dataset):

    def __init__(self, dates, start_date, end_date, regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, Show_file_name=True,validation=False):
        print("=> ACCESS_S2 & AWAP loading")
        print("=> from " + start_date.strftime("%Y/%m/%d") + " to " + end_date.strftime("%Y/%m/%d") + "")
        # self.file_ACCESS_dir = "/scratch/iu60/rw6151/access_40_7_masked/"
        # self.file_AWAP_dir = "/scratch/iu60/rw6151/Split_AWAP_masked_total/"
        self.file_ACCESS_dir = "/scratch/iu60/xs5813/Processed_data_train/"
        self.file_AWAP_dir = "/scratch/iu60/xs5813/Awap_data_bigger/"

        # self.regin = regin
        self.start_date = start_date
        self.end_date = end_date

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.leading_time_we_use = 6

        self.ensemble = ['e01', 'e02', 'e03','e04', 'e05', 'e06','e07', 'e08', 'e09']

        # self.dates = date_range(start_date, end_date)
        # self.initial_dates = self.get_initial_date(self.file_ACCESS_dir)
        # random.shuffle(self.initial_dates)
        # self.train_dates = self.initial_dates[:int((len(self.initial_dates)+1)*.80)] # 80% training
        # self.test_dates = self.initial_dates[int((len(self.initial_dates)+1)*.80):] # 20% test
        # self.train_files = self.get_files_on_date(self.file_ACCESS_dir, self.train_dates)
        # self.test_files = self.get_files_on_date(self.file_ACCESS_dir, self.test_dates)
        if validation:
            self.filename_list = self.get_files_on_date_validation(self.file_ACCESS_dir, dates)
        else:
            self.filename_list = self.get_files_on_date(self.file_ACCESS_dir, dates)
        
        # self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir + "pr/daily/")
            print("no file or no permission")

        # _, _, date_for_AWAP, time_leading = self.filename_list[0]
        if Show_file_name:
            print("we use these files for train or test:", self.filename_list)
        # if shuffle:
        #     random.shuffle(self.filename_list)

        # data_high = read_awap_data_fc_get_lat_lon(self.file_AWAP_dir, date_for_AWAP)
        # print("data_high")
        # print(data_high)
        # self.lat = data_high[1]
        # print(self.lat)
        # self.lon = data_high[1]
        # print(self.lon)
        # sshape = (79, 94)

    def __len__(self):
        return len(self.filename_list)

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

    def get_initial_date(self, rootdir):
        '''
        This function is used to extract the date that we plan to use in training
        '''
        _dates = []
        for date in self.dates:
            access_path = rootdir + "e09/da_pr_" + date.strftime("%Y%m%d") + "_e09.nc"
            if os.path.exists(access_path):
                _dates.append(date)
        return _dates

    def get_files_on_date(self, rootdir, _dates):
        '''
        find the files from 9 ensembles on specific date
        '''
        _files = []
        lead_time = np.arange(1, self.leading_time_we_use+1)
        random.shuffle(lead_time)
        print('lead time is:', lead_time)
        newleadtime = [int(i) for i in list(lead_time)]
        for i in newleadtime:
            for date in _dates:
                random.shuffle(self.ensemble)
                print('random ensemble members:', self.ensemble)
                for en in self.ensemble:
                    filename = rootdir + en + "/da_pr_" + date.strftime("%Y%m%d") + "_" + en + ".nc"
                    print('filename: ', filename)
                    if os.path.exists(filename):
                        path = [en]
                        AWAP_date = date + timedelta(i)
                        path.append(date)
                        path.append(AWAP_date)
                        path.append(i)
                        _files.append(path)
        return _files
    def get_files_on_date_validation(self, rootdir, _dates):
        '''
        find the files from 9 ensembles on specific date, adjust date selection based on leading time
        '''
        _files = []

        for i in np.arange(1, self.leading_time_we_use + 1):  # 假设self.leading_time_we_use为7
            
            # 根据leading time调整selected_dates的数量
            if i == 5:
                selected_dates = random.sample(_dates, int(len(selected_dates) * 0.8))
            elif i == 6:
                selected_dates = random.sample(_dates, int(len(selected_dates) * 0.6))
            elif i == 7:
                selected_dates = random.sample(_dates, int(len(selected_dates) * 0.4))
            else:
                selected_dates = _dates

            for date in selected_dates:
                random.shuffle(self.ensemble)  # 随机排列ensemble成员
                for en in self.ensemble:
                    filename = f"{rootdir}{en}/da_pr_{date.strftime('%Y%m%d')}_{en}.nc"
                    if os.path.exists(filename):
                        path = [en]
                        AWAP_date = date + timedelta(i)
                        path.append(date)
                        path.append(AWAP_date)
                        path.append(i)
                        _files.append(path)
        return _files


    def get_filename_with_time_order(self, rootdir):
        '''
        get filename first and generate label ,one different w
        Check whether the date in other ensemble folders exist in e09.
        If the date exists in e09 folder, then add the corresponding data into filename_list.

        return: e_num, data, AWAP_date, leading time
        '''

        _files = []
        for en in self.ensemble:
            for date in self.dates:
                access_path = rootdir + "e09" + "/da_pr_" + date.strftime("%Y%m%d") + "_" + "e09" + ".nc"
                #                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date == self.end_date and i == 1:
                            break
                        path = [en]
                        AWAP_date = date + timedelta(i)
                        path.append(date)
                        path.append(AWAP_date)
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
        # en, access_date, awap_date, time_leading = self.train_files[idx]
        lr = read_access_data(self.file_ACCESS_dir, en, access_date, time_leading, "pr")

        hr = read_awap_data(self.file_AWAP_dir, awap_date)

        return lr, hr, en, access_date.strftime("%Y%m%d"), awap_date.strftime("%Y%m%d"), time_leading

def CRPS(y_true, y_pred):
    # Initialize prediction matrices
    y_pred = np.array(y_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    # Generate 10 predicted values based on Gamma distribution
    
    # Remove border pixels
    forecasts = np.expm1(y_pred * 7)
    # Calculate CRPS
    crps = ps.crps_ensemble(y_true, np.transpose(forecasts, (1, 2, 0)))

    return crps.mean().item()

class ACCESS_AWAP_GAN_crps(Dataset):

    def __init__(self, start_date, end_date, regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, args=None):
        print("=> ACCESS_S2 & AWAP loading")
        print("=> from " + start_date.strftime("%Y/%m/%d") + " to " + end_date.strftime("%Y/%m/%d") + "")
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.file_AWAP_dir = args.file_AWAP_dir

        # self.regin = regin
        self.start_date = start_date
        self.end_date = end_date

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.leading_time_we_use = args.leading_time_we_use

        self.ensemble = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06']

        self.dates = date_range(start_date, end_date)

        self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir + "pr/daily/")
            print("no file or no permission")

        _, _, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        # data_high = read_awap_data_fc_get_lat_lon(self.file_AWAP_dir, date_for_AWAP)
        # print("data_high")
        # print(data_high)
        # self.lat = data_high[1]
        # print(self.lat)
        # self.lon = data_high[1]
        # print(self.lon)
        # sshape = (79, 94)

    def __len__(self):
        return len(self.filename_list)

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

    def get_initial_date(self, rootdir):
        '''
        This function is used to extract the date that we plan to use in training
        '''
        _dates = []
        for date in self.dates:
            access_path = rootdir + "e09/da_pr_" + date.strftime("%Y%m%d") + "_e09.nc"
            if os.path.exists(access_path):
                _dates.append(date)
        return _dates

    def get_files_on_date(self, rootdir, _dates):
        '''
        find the files from 9 ensembles on specific date
        '''
        _files = []
        for date in _dates:
            for en in self.ensemble:
                filename = rootdir + en + "/da_pr_" + date.strftime("%Y%m%d") + "_" + en + ".nc"
                print('filename: ', filename)
                if os.path.exists(access_path):
                    path = [en]
                    AWAP_date = date + timedelta(i)
                    path.append(date)
                    path.append(AWAP_date)
                    path.append(i)
                    _files.append(path)
        return _files

    def get_filename_with_time_order(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(1, self.leading_time_we_use + 1):
                for en in self.ensemble:
                    access_path = rootdir + en + "/da_pr_" + date.strftime("%Y%m%d") + "_" + en + ".nc"
                    # print(access_path)
                    if os.path.exists(access_path):
                        if date == self.end_date and i == 1:
                            break
                        path = [en]
                        AWAP_date = date + timedelta(i)
                        path.append(date)
                        path.append(AWAP_date)
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

        lr = read_access_data(self.file_ACCESS_dir, en, access_date, time_leading, "pr")

        hr = read_awap_data(self.file_AWAP_dir, awap_date)

        return lr, hr, awap_date.strftime("%Y%m%d"), time_leading


def read_awap_data(root_dir, date_time):
    filename = root_dir + date_time.strftime("%Y-%m-%d") + ".nc"
    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)
    # rescale to [0,1]
    var = dataset.isel(time=0)['precip'].values
    var = (np.log1p(var)) / 7
    var = var[np.newaxis, :, :].astype(np.float32)  # CxLATxLON
    dataset.close()
    return var


def read_access_data(root_dir, en, date_time, leading, var_name="pr"):
    filename = root_dir + en + "/da_pr_" + date_time.strftime("%Y%m%d") + "_" + en + ".nc"
    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)

    # rescale to [0,1]
    var = dataset.isel(time=leading)['pr'].values * 86400
    var = np.clip(var, 0, 1000)
    var = (np.log1p(var)) / 7

    var = cv2.resize(var, (33, 51), interpolation=cv2.INTER_CUBIC)
    var = var[np.newaxis, :, :].astype(np.float32)  # CxLATxLON
    dataset.close()
    return var
