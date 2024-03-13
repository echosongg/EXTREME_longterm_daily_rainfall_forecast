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

def AWAPcalpercentile(startyear, endyear, i, p_value):
    filepath = "/scratch/iu60/yl3101/AGCD_mask_data/" 
    pr_value = []
    for file in os.listdir(filepath):
        if startyear <= int(file[:4]) <= endyear:

            dataset = xr.open_dataset(filepath + file)
            dataset = dataset.fillna(0)
            var = dataset.isel(time=i)['precip'].values
            var = (np.log1p(var)) / 7
            pr_value.append(var)
            dataset.close()
    np_pr_value = np.array(pr_value)
    return np.percentile(np_pr_value, p_value, axis=0)
print(AWAPcalpercentile(1981, 2010, 1, 90))