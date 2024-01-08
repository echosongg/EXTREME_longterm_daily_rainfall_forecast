import os
import random
from datetime import date, timedelta
import torch
from torchvision import transforms
import time
import numpy as np
import xarray as xr
import cv2
from torch.utils.data import Dataset
import xskillscore as xs


### COMMON FUNCTIONS ###


# def dumplicatearray(data, num_repeats):
#     return np.dstack([data] * num_repeats)


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

def CRPS(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    crps = xs.crps_ensemble(target_data, diff)

    return crps

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


def days_in_month(month: int, year: int) -> int:
        """
        Returns the number of days in the specified month.

        Parameters:
        - month (int): The month.
        - year (int): The year.

        Returns:
        - int: The number of days in the specified month.
        """
        if month in [1, 3, 5, 7, 8, 10, 12]:
            return 31
        elif month in [4, 6, 9, 11]:
            return 30
        elif year % 4 == 0 and (year % 100 != 0 or year % 400 == 0): # Leap year
            return 29
        else:
            return 28

def increment_months(datetime: str, leadtime: int) -> str:
    """
    Increments the specified date by one month.

    Parameters:
    - datetime (str): The date in the format 'YYYY-MM-DD'.
    - leadtime (int): The number of months to increment by.

    Returns:
    - str: The incremented date in the format 'YYYY-MM-DD'.
    """
    # Get the year, month and day
    year, month, day = datetime.split('-')
    year = int(year)
    month = int(month)
    day = int(day)

    # Increment month
    new_month = month + leadtime
    if new_month > 12:
        year += 1
        new_month -= 12

    # Day can be 1, 6, 11, 16, 21 or the last 8 days of the month 
    # To account for months having a different number of days we remap the day
    # We map 1 -> 1, 6 -> 6, 11 -> 11, 16 -> 16, 21 -> 21 and then ith last day of old month -> ith last day of new month
    if day > 21:
        day_offset = days_in_month(month, year) - day
        new_day = days_in_month(new_month, year) - day_offset
    else:
        new_day = day

    return f"{year}-{str(new_month).zfill(2)}-{str(new_day).zfill(2)}"


def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

'''在 read_access_data 函数中的应用: 在这个函数中，您使用 dataset.isel(time=leading)['pr'].values 从 ACCESS 数据集中选取了具有特定 leading time 的预测数据。
这意味着如果 leading 参数为 1，则函数将返回模型启动后第二个时间点（考虑到索引从 0 开始）的预测数据。'''
class ACCESS_AWAP_GAN(Dataset):
    #这要改
    def __init__(self, start_date, end_date, regin="AUS", lr_transform=None, hr_transform=None, shuffle=True,
                 access_dir = "/scratch/iu60/xs5813/Processed_data/",
                    awap_dir = "/scratch/iu60/xs5813/Awap_pre_data/"):
        # Data locations
        self.file_ACCESS_dir = access_dir 
        self.file_AWAP_dir = awap_dir

        # self.regin = regin0
        self.start_date = start_date
        self.end_date = end_date

        # Transforms
        self.lr_transform = lr_transform if lr_transform else transforms.Compose([transforms.ToTensor()])
        self.hr_transform = hr_transform if hr_transform else transforms.Compose([transforms.ToTensor()])

        # Data
        #这里可以改一下
        self.leading_time_we_use = 1
        self.ensemble = ['e01','e02','e03']
        self.dates = date_range(start_date, end_date)

        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir + "pr/daily/")
            print("no file or no permission")
        
        self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)

        assert len(self.filename_list) > 0, "No data found in " + self.file_ACCESS_dir

        if shuffle:
            random.shuffle(self.filename_list)

    def __len__(self):
        return len(self.filename_list)

    def get_filename_with_time_order(self, rootdir):
        """
            Get AWAP date (hr), ACCESS date (lr), ensemble and leading time tuples in time order.

            Args:
                rootdir (str): root directory of ACCESS data
            
            Returns:
                file_list (list):  list of tuples of AWAP date (hr), ACCESS date (lr), ensemble and leading time
        """
        file_list = []
        for ens in self.ensemble:
            for date in self.dates:
                access_path = rootdir + ens + "/" + date.strftime("%Y-%m-%d") + ".nc"
                if os.path.exists(access_path):
                    for lead_time in range(self.leading_time_we_use + 1): # 0 - leading_time_we_use
                        
                        hr_lr_data = (
                            ens, # Ensemble
                            date, # ACCESS date
                            date.fromisoformat(
                                increment_months(date.strftime("%Y-%m-%d"), lead_time)
                            ), # AWAP date
                            lead_time # Leading time used for ACCESS data
                        )
                        file_list.append(hr_lr_data)

        return file_list

    def __getitem__(self, idx):
        """
            Given an index into the file name list, returns the paired preprocessed lr and hr data.

            Args:
                idx (int): Index into the file name list
            
            Returns:
                lr (torch.Tensor): Low-resolution data
                hr (torch.Tensor): High-resolution data
                date (str): Date of the high-resolution data
                time_leading (int): Leading time of the low-resolution data
        """

        # read_data filemame[idx]
        en, access_date, awap_date, time_leading = self.filename_list[idx]

        lr = read_access_data(self.file_ACCESS_dir, en, access_date, time_leading)
        hr = read_awap_data(self.file_AWAP_dir, awap_date)
        return lr, hr, awap_date.strftime("%Y%m%d"), time_leading



def read_awap_data(root_dir, date_time):

    filename = root_dir + date_time.strftime("%Y-%m-%d") + ".nc"
    dataset = xr.open_dataset(filename)
    
    var = dataset['pr'].values
    #print("AWAP data shape (before processing):", var.shape)
    #这里除以4是干啥
    var = (np.log1p(var)) / 4 # log1p(x) to fix skew in distribution, /4 to scale roughly to [0,1]
    var = var[np.newaxis, :, :].astype(np.float32)  # CxLATxLON
    
    dataset.close()
    return var



def read_access_data(root_dir, en, date_time, leading):
    """
        Reads ACCESS data from netcdf file, applies preprocessing steps (log1p, scaling) and returns data as numpy array.

        Args:
            root_dir (str): root directory of ACCESS data
            en (str): ensemble member
            date_time (datetime.date): date of ACCESS data to be read
            leading (int): leading time of ACCESS data to be read
        
        Returns:
            var (np.ndarray): numpy array of ACCESS data
    """

    # Get the filename of the netcdf file
    filename = root_dir + en + "/" + date_time.strftime("%Y-%m-%d") + ".nc"
    dataset = xr.open_dataset(filename)

    # rescale to [0,1]
    var = dataset.isel(time=leading)['pr'].values
    #print("ACCESS data shape (before processing):", var.shape)
    var = (np.log1p(var)) / 4 # log1p(x) to fix skew in distribution, /4 to scale roughly to [0,1]
    
    var = var[np.newaxis, :, :].astype(np.float32)  # CxLATxLON

    dataset.close()
    return var

'''def read_awap_data(root_dir, date_time):
    """
        Reads AWAP data from netcdf file, applies preprocessing steps (log1p, scaling) and returns data as numpy array.

        Args: 
            root_dir (str): root directory of AWAP data
            date_time (datetime.date): date of AWAP data to be read
        
        Returns:
            var (np.ndarray): numpy array of AWAP data
    """

    # Get the filename of the netcdf file
    filename = root_dir + date_time.strftime("%Y-%m") + ".nc"
    dataset = xr.open_dataset(filename)
    
    var = dataset['precip'].values
    var = (np.log1p(var)) / 4 # log1p(x) to fix skew in distribution, /4 to scale roughly to [0,1]
    var = var[np.newaxis, :, :].astype(np.float32)  # CxLATxLON
    dataset.close()

    return var'''