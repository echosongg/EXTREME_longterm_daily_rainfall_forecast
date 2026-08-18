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
import properscoring as ps


### COMMON FUNCTIONS ###
def dumplicatearray(data, num_repeats):
    return np.dstack([data] * num_repeats)

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

# def crps_loss(batch_H, bg_output, epsilon=1e-6,  lambda_val=0.1):
#     """
#     Calculate the log-likelihood loss for a Bernoulli-Gamma distribution, with an option to reduce by mean.

#     Args:
#         batch_H: The actual rainfall data.
#         p_pred: The predicted probability of rainfall from the neural network.
#         alpha_pred: The predicted shape parameter of the Gamma distribution.
#         beta_pred: The predicted scale parameter of the Gamma distribution.
#         epsilon: A small value for numerical stability.
#         reduce: If True, returns the mean of the losses, else returns the losses for each sample.

#     Returns:
#         The calculated loss, either reduced by mean or as individual losses per sample.
#     """
#     p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)  # 下雨概率, 形状 [3, 1, 128, 128]
#     alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)  # gamma shape, 形状 [3, 1, 128, 128]
#     beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)  # gamma scale, 形状 [3, 1, 128, 128]
#     p_true = (batch_H > 0).float()
#     term1 = -(1 - p_true) * torch.log(1 - p_pred + epsilon)
#     term2 = p_true * (-torch.log(p_pred+epsilon) + lambda_val * gamma_crps(alpha_pred, beta_pred, batch_H))
#     loss = term1 + term2
#     return torch.mean(loss)

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

def generate_random_sample(bg_output):
    # 下雨概率, 形状 [3, 1, 128, 128]
    p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)
    # 使用伯努利分布根据 p_pred 生成 0 或 1
    random_sample = torch.bernoulli(p_pred)

    # gamma 形状参数, 形状 [3, 1, 128, 128]
    alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)
    # gamma 比率参数, 形状 [3, 1, 128, 128]
    beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)

    # 随机样本与 alpha 和 beta 的乘积
    output = random_sample * (alpha_pred * beta_pred)
    return output

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

def CRPS_from_distribution(p_pred, alpha_pred, beta_pred, y_true, shave_border=4):
    # Initialize prediction matrices
    p_pred = np.array(p_pred, dtype=np.float32)
    alpha_pred = np.array(alpha_pred, dtype=np.float32)
    beta_pred = np.array(beta_pred, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.float32)
    forecasts = np.zeros((30, *p_pred.shape))  # Assuming p_pred, alpha_pred, beta_pred shapes are the same
    
    # Generate 10 predicted values based on Gamma distribution
    for i in range(30):
        # For each pixel point, generate a prediction value based on Gamma distribution parameters
        is_rain = np.random.binomial(1, p_pred)  # Use binomial distribution to determine if there's rain
        rain_amount = np.random.gamma(alpha_pred, beta_pred)  # Generate rain amount from Gamma distribution
        forecasts[i] = is_rain * rain_amount  # If no rain, rain amount is 0
    
    # Remove border pixels
    forecasts = np.expm1(forecasts * 7)
    forecasts = np.clip(forecasts, None, 300)
    print("y_true value",y_true)
    # Calculate CRPS
    crps = ps.crps_ensemble(y_true, np.transpose(forecasts, (1, 2, 3, 0)))

    return crps.mean().item()

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



def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

'''在 read_access_data 函数中的应用: 在这个函数中，您使用 dataset.isel(time=leading)['pr'].values 从 ACCESS 数据集中选取了具有特定 leading time 的预测数据。
这意味着如果 leading 参数为 1，则函数将返回模型启动后第二个时间点（考虑到索引从 0 开始）的预测数据。'''
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
                 hr_transform=None, shuffle=True, Show_file_name=True,validation = False):
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
        lead_time = np.arange(1, self.leading_time_we_use)
        random.shuffle(lead_time)
        newleadtime = [int(i) for i in list(lead_time)]
        for i in newleadtime:
            for date in _dates:
                random.shuffle(self.ensemble)
                print('random ensemble members:', self.ensemble)
                for en in self.ensemble:
                    filename = rootdir + en + "/da_pr_" + date.strftime("%Y%m%d") + "_" + en + ".nc"
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
        lead_time = np.arange(1, self.leading_time_we_use)
        random.shuffle(lead_time)
        newleadtime = [int(i) for i in list(lead_time)]
        if not isinstance(_dates, list):
            _dates = list(_dates)
        for i in newleadtime:  # 假设self.leading_time_we_use为7

            selected_dates = _dates  
            # 根据leading time调整selected_dates的数量
            if i == 5:
                selected_dates = random.sample(_dates, int(len(selected_dates) * 0.8))
            elif i == 6:
                selected_dates = random.sample(_dates, int(len(selected_dates) * 0.6))
            elif i == 7:
                selected_dates = random.sample(_dates, int(len(selected_dates) * 0.4))

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


def read_awap_data(root_dir, date_time):
    filename = root_dir + date_time.strftime("%Y-%m-%d") + ".nc"
    dataset = xr.open_dataset(filename)
    dataset = dataset.fillna(0)
    # rescale to [0,1]
    var = dataset.isel(time=0)['precip'].values
    var[var <= 0.1] = 0
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
