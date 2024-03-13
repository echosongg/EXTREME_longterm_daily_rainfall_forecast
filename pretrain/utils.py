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
import scipy.stats
import scipy.integrate


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

def log_loss(batch_H, p_pred, alpha_pred, beta_pred, epsilon=1e-6):
    """
    Calculate the log-likelihood loss for a Bernoulli-Gamma distribution.

    Args:
        batch_H: The actual rainfall data.
        p_pred: The predicted probability of rainfall from the neural network.
        alpha_pred: The predicted shape parameter of the Gamma distribution.
        beta_pred: The predicted scale parameter of the Gamma distribution.
        epsilon: A small value for numerical stability.

    Returns:
        The calculated loss.
    """
    print("start log loss")
    batch_H = batch_H.detach().cpu().numpy()
    p_pred = p_pred.detach().cpu().numpy()
    alpha_pred = alpha_pred.detach().cpu().numpy()
    beta_pred = beta_pred.detach().cpu().numpy()

    # Here we assume rainfall if the predicted probability is less than 0.5
    p_true = (batch_H > 0).astype(float)
    term1 = (1 - p_true) * np.log(1 - p_pred + epsilon)
    term2 = p_true * (np.log(p_pred + epsilon) + (alpha_pred - 1) * np.log(batch_H + epsilon) - batch_H / (beta_pred + epsilon) - alpha_pred * np.log(beta_pred + epsilon) - scipy.special.gammaln(alpha_pred + epsilon))
    loss = term1 + term2
    return -np.mean(loss)

def log_loss2(batch_H, bg_output, epsilon=1e-6, reduce=True):
    """
    Calculate the log-likelihood loss for a Bernoulli-Gamma distribution, with an option to reduce by mean.

    Args:
        batch_H: The actual rainfall data.
        p_pred: The predicted probability of rainfall from the neural network.
        alpha_pred: The predicted shape parameter of the Gamma distribution.
        beta_pred: The predicted scale parameter of the Gamma distribution.
        epsilon: A small value for numerical stability.
        reduce: If True, returns the mean of the losses, else returns the losses for each sample.

    Returns:
        The calculated loss, either reduced by mean or as individual losses per sample.
    """
    p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)  # 下雨概率, 形状 [3, 1, 128, 128]
    alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)  # gamma shape, 形状 [3, 1, 128, 128]
    beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)  # gamma scale, 形状 [3, 1, 128, 128]

    p_true = (batch_H > 0).float()
    term1 = (1 - p_true) * torch.log(1 - p_pred + epsilon)
    term2 = p_true * (torch.log(p_pred + epsilon) + (alpha_pred - 1) * torch.log(batch_H + epsilon) - batch_H / (beta_pred + epsilon) - alpha_pred * torch.log(beta_pred + epsilon) - torch.lgamma(alpha_pred + epsilon))
    loss = term1 + term2

    if reduce:
        return -torch.mean(loss)
    else:
        return -loss

from torch.distributions import Gamma
from torch import special as torch_special

def check_tensor(tensor, name):
    if torch.isinf(tensor).any():
        print(f"{name} contains inf. Stats: min={tensor.min()}, max={tensor.max()}")
    if torch.isnan(tensor).any():
        print(f"{name} contains nan. Stats: min={tensor.min()}, max={tensor.max()}")


def generate_x_points_per_pixel(y, steps=10, epsilon=1e-6):
    batch_size, channels, height, width = y.shape
    device = y.device
    y_min = torch.zeros_like(y) + epsilon  # 确保起始点为一个小的正数
    # 由于y中可能包含0，调整结束点以确保所有x_points都是正的
    y_max = y + epsilon
    relative_positions = torch.linspace(0, 1, steps=steps, device=device).view(1, 1, 1, 1, steps)
    y_expanded = y_min.unsqueeze(-1) + (y_max - y_min).unsqueeze(-1) * relative_positions  # 使用调整后的范围
    x_points = y_expanded  # 调整范围
    return x_points


def gamma_crps(alpha, beta, y, steps=10, epsilon=1e-6):

    device = alpha.device
    beta = 1/beta
    x_points = generate_x_points_per_pixel(y, steps=steps)

    # 扩充alpha和beta以匹配x_points的形状
    alpha_expanded = alpha.unsqueeze(-1).expand(*alpha.shape, steps)
    beta_expanded = beta.unsqueeze(-1).expand(*beta.shape, steps)
    print("min alpha",torch.min(alpha_expanded))
    print("min beta",torch.min(beta_expanded))
    print("min y",torch.min(y))

    # 计算PDF值
    gamma_dist = Gamma(alpha_expanded, beta_expanded)
    log_probs = gamma_dist.log_prob(x_points)
    pdf_values = torch.exp(log_probs)
    print("pdf_values min", torch.min(pdf_values))
    # 计算每个小段的dx
    dx = x_points[..., 1:] - x_points[..., :-1]

    # 使用梯形法则计算CDF的近似值
    trapezoids_area = (pdf_values[..., :-1] + pdf_values[..., 1:]) * dx / 2.0
    F_alpha_beta = torch.sum(trapezoids_area, dim=-1)
    print("F_alpha_beta", torch.min(F_alpha_beta))

    # 对于alpha + 1重复相同的过程计算F_alpha_plus_1_beta
    gamma_dist_alpha_plus_1 = Gamma(alpha_expanded + 1, beta_expanded)
    log_probs_alpha_plus_1 = gamma_dist_alpha_plus_1.log_prob(x_points)
    pdf_values_alpha_plus_1 = torch.exp(log_probs_alpha_plus_1)
    print("pdf_values_alpha_plus_1 min", torch.min(pdf_values_alpha_plus_1))
    trapezoids_area_alpha_plus_1 = (pdf_values_alpha_plus_1[..., :-1] + pdf_values_alpha_plus_1[..., 1:]) * dx / 2.0
    F_alpha_plus_1_beta = torch.sum(trapezoids_area_alpha_plus_1, dim=-1)
    # 计算CRPS的各项
    print("F_alpha_plus_1_beta", torch.min(F_alpha_plus_1_beta))
    first_term = y * (2 * F_alpha_beta - 1) #negative
    second_term = (alpha / (beta+epsilon)) * (2 * F_alpha_plus_1_beta - 1) #positive
    third_term = 2 * alpha / (beta+epsilon) * torch.exp(torch.lgamma(alpha + 0.5)) / (torch.sqrt(torch.tensor(np.pi)) * torch.exp(torch.lgamma(alpha + 1))+epsilon)
    # 计算CRPS
    crps = first_term - second_term - 0.5 * torch.exp(third_term)
    print("raw crps max", torch.max(crps))
    print("raw crps min", torch.min(crps))
    print("raw crps mean", torch.mean(crps))

    return crps

def crps_loss(batch_H, bg_output, epsilon=1e-6,  lambda_val=0.1):
    """
    Calculate the log-likelihood loss for a Bernoulli-Gamma distribution, with an option to reduce by mean.

    Args:
        batch_H: The actual rainfall data.
        p_pred: The predicted probability of rainfall from the neural network.
        alpha_pred: The predicted shape parameter of the Gamma distribution.
        beta_pred: The predicted scale parameter of the Gamma distribution.
        epsilon: A small value for numerical stability.
        reduce: If True, returns the mean of the losses, else returns the losses for each sample.

    Returns:
        The calculated loss, either reduced by mean or as individual losses per sample.
    """
    p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)  # 下雨概率, 形状 [3, 1, 128, 128]
    alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)  # gamma shape, 形状 [3, 1, 128, 128]
    beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)  # gamma scale, 形状 [3, 1, 128, 128]
    p_true = (batch_H > 0).float()
    term1 = -(1 - p_true) * torch.log(1 - p_pred + epsilon)
    term2 = p_true * (-torch.log(p_pred+epsilon) + lambda_val * gamma_crps(alpha_pred, beta_pred, batch_H))
    loss = term1 + term2
    return torch.mean(loss)

def generate_sample(bg_output):
    p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)  # 下雨概率, 形状 [3, 1, 128, 128]
    p_pred = (p_pred > 0.5).float()
    alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)  # gamma shape, 形状 [3, 1, 128, 128]
    beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)  
    #p_smooth = torch.sigmoid((rain_prob - 0.5) * tau)
    return p_pred * ((alpha_pred) * beta_pred)
def generate_3_channels(bg_output):
    p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)  # 下雨概率, 形状 [3, 1, 128, 128]
    alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)  # gamma shape, 形状 [3, 1, 128, 128]
    beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)  
    return p_pred, alpha_pred, beta_pred

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



def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

'''在 read_access_data 函数中的应用: 在这个函数中，您使用 dataset.isel(time=leading)['pr'].values 从 ACCESS 数据集中选取了具有特定 leading time 的预测数据。
这意味着如果 leading 参数为 1，则函数将返回模型启动后第二个时间点（考虑到索引从 0 开始）的预测数据。'''
class ACCESS_AWAP_GAN(Dataset):
    def __init__(self, start_date, end_date, region="AUS", lr_transform=None, hr_transform=None, shuffle=True,
                 access_dir="/scratch/iu60/xs5813/Processed_data/", awap_dir="/scratch/iu60/xs5813/Awap_pre_data/",
                 summer=True):  # 添加summer参数
        # Data locations
        print("=> ACCESS_S1 & AWAP loading")
        print("=> from " + start_date.strftime("%Y/%m/%d") + " to " + end_date.strftime("%Y/%m/%d") + "")
        self.file_ACCESS_dir = access_dir
        self.file_AWAP_dir = awap_dir

        self.start_date = start_date
        self.end_date = end_date
        self.summer = summer  # 保存summer到实例变量

        # Transforms
        self.lr_transform = lr_transform if lr_transform else transforms.Compose([transforms.ToTensor()])
        self.hr_transform = hr_transform if hr_transform else transforms.Compose([transforms.ToTensor()])

        # Data
        self.leading_time_we_use = 7
        self.ensemble = ['e01', 'e02', 'e03']
        self.dates = date_range(start_date, end_date)

        if summer:  # 如果summer为True，则过滤夏季月份
            self.dates = [date for date in self.dates if date.month in [10, 11, 12, 1, 2, 3]]

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
                    for lead_time in range(1, self.leading_time_we_use + 1): # 1 - leading_time_we_use
                        
                        hr_lr_data = (
                            ens, # Ensemble
                            date, # ACCESS date
                            date + timedelta(lead_time), # AWAP date
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
        #utils lr shape (1, 21, 19)
        #utils hr shape (1, 151, 226)
        return lr, hr, awap_date.strftime("%Y%m%d"), time_leading



def read_awap_data(root_dir, date_time):

    filename = root_dir + date_time.strftime("%Y-%m-%d") + ".nc"
    #print("awap filename",filename)
    dataset = xr.open_dataset(filename)
    
    var = dataset['pr'].values
    #print(f"AWAP data stats - Max: {np.max(var)}, Min: {np.min(var)}")
    #print("AWAP data shape (before processing):", var.shape)
    #这里除以4是干啥
    var = (np.log1p(var)) / 7 # log1p(x) to fix skew in distribution, /4 to scale roughly to [0,1]
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
    dataset = dataset.fillna(0)
    # rescale to [0,1]
    var = dataset.isel(time=leading)['pr'].values
    var = np.clip(var, 0, 1000)
    #print(f"ACCESS data stats - Max: {np.max(var)}, Min: {np.min(var)}")
    #print("ACCESS data shape (before processing):", var.shape)
    var = (np.log1p(var)) / 7 # log1p(x) to fix skew in distribution, /4 to scale roughly to [0,1]

    var = var[np.newaxis, :, :].astype(np.float32)  # CxLATxLON
    dataset.close()
    return var