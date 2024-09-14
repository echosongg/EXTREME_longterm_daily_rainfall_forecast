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
def compute_pit(forecast, y_true):
    '''
    forecast: (ensemble, H, W) -> (9, 413, 267)
    y_true: (H, W) -> (413, 267)
    '''
    ensemble, H, W = forecast.shape
    
    # 初始化 PIT 数组
    PIT = np.zeros((H, W))
    
    # 对每个像素点计算 PIT 值
    for h in range(H):
        for w in range(W):
            # 获取当前像素点的九个预测值
            predictions = forecast[:, h, w]
            
            # 将预测值按升序排序
            sorted_predictions = np.sort(predictions)
            
            # 找到真实值 y_true[h, w] 在排序预测值中的位置
            if y_true[h, w] < sorted_predictions[0]:
                PIT[h, w] = 0  # 真实值小于最小预测值
            elif y_true[h, w] > sorted_predictions[-1]:
                PIT[h, w] = 1  # 真实值大于最大预测值
            else:
                # 找到真实值在排序预测值中的位置并进行线性插值
                for i in range(ensemble - 1):
                    if sorted_predictions[i] <= y_true[h, w] <= sorted_predictions[i + 1]:
                        # 线性插值计算 PIT
                        PIT[h, w] = (i + (y_true[h, w] - sorted_predictions[i]) / 
                                     (sorted_predictions[i + 1] - sorted_predictions[i])) / (ensemble + 1)
                        break
    return PIT

def alpha_index(forecast, y_true):
    '''
    forecast: (ensemble, H, W) -> (9, 413, 267)
    y_true: (H, W) -> (413, 267)
    '''
    # 计算 PIT 值
    PIT = compute_pit(forecast, y_true)
    
    # 排序 PIT 值
    sorted_PIT = np.sort(PIT, axis=None)  # 将整个数组展平后排序
    
    # 样本数量
    n = len(sorted_PIT)
    
    # 计算理想的均匀分布期望值
    t_values = np.arange(1, n + 1)  # t 从 1 到 n
    expected_values = t_values / (n + 1)
    
    # 计算 alpha 指数
    alpha = 1 - (2 / n) * np.sum(sorted_PIT - expected_values)
    
    return alpha
def mae(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens - hr)).sum(axis=0) / ens.shape[0]


def bias(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return (ens - hr).sum(axis=0) / ens.shape[0]


def bias_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.median(ens, axis=0) - hr


def bias_relative(ens, hr, constant=1):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    constant: relative constant
    '''
    return (np.mean(ens, axis=0) - hr) / (constant + hr)


def bias_relative_median(ens, hr, constant=1):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    constant: relative constant
    '''
    return (np.median(ens, axis=0) - hr) / (constant + hr)


def rmse(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt(((ens - hr) ** 2).sum(axis=(0)) / ens.shape[0])


def mae_mean(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens.mean(axis=0) - hr))


def mae_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((np.median(ens, axis=0) - hr))

def calAWAPprob(AWAP_data, percentile):
    ''' 
    input: AWAP_data is  413 * 267
            percentile size is 413 * 267
    return: A probability matrix which size is 413 * 267 indicating the probability of the values in ensemble forecast 
    is greater than the value in the same pixel in percentile matrix

    '''

    return (AWAP_data > percentile) * 1

def calforecastprob(forecast, percentile):
    ''' 
    input: forecast is  9 * 413 * 267
            percentile size is 413 * 267
    return: A probability matrix which size is 413 * 267 indicating the probability of the values in ensemble forecast 
    is greater than the value in the same pixel in percentile matrix

    '''
    prob_matrix = (forecast > percentile)
    return np.mean(prob_matrix, axis = 0)

def calAWAPdryprob(AWAP_data, percentile):

    return (AWAP_data >= percentile) * 1

def calforecastdryprob(forecast, percentile):

    prob_matrix = (forecast >= percentile)
    return np.mean(prob_matrix, axis = 0)   

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
    percentile_95 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    percentile_99 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    percentile_995 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99.5)
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

        mean_bias_model = []
        mean_bias_median_model = []
        mean_crps_model = []
        mae_score = []
        Brier_0 = []
        Brier95 = []
        Brier99 = []
        Brier995 = []
        Brier999 = []
        heavy30 = []
        alpha = []
        mae_median_score_model = []
        mean_bias_relative_model = []
        mean_bias_relative_model_half = []
        mean_bias_relative_model_1 = []
        mean_bias_relative_model_2 = []
        mean_bias_relative_model_2d9 = []
        mean_bias_relative_model_3 = []
        mean_bias_relative_model_5 = []

        for batch, (pr, dem, hr, en, data_time, idx) in enumerate(test_data):

            with torch.set_grad_enabled(False):

                #                 sr = net(pr)
                sr_np = pr.cpu().numpy()
                hr_np = hr.cpu().numpy()

                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(
                        sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b = np.squeeze(hr_np[i * args.ensemble])

                    # for each ensemble member
                    bias_QM = bias(a, b)
                    bias_median_qm = bias_median(a,b)
                    skill_QM = ps.crps_ensemble(b, np.transpose(a, (1, 2, 0)))
                    QM_mae = mae_mean(a,b)
                    QM_alpha = alpha_index(a, b)
                    prob_awap0 = calAWAPdryprob(b, 0.1)
                    prob_forecast_0 = calforecastdryprob(a, 0.1)
                    Brier_0.append((prob_awap0 - prob_forecast_0) ** 2)
                    prob_AWAP_95 = calAWAPprob(b, percentile_95)
                    prob_forecast_95 = calforecastprob(a, percentile_95)
                    Brier95.append((prob_AWAP_95 - prob_forecast_95) ** 2)
                    prob_AWAP_99 = calAWAPprob(b, percentile_99)
                    prob_forecast_99 = calforecastprob(a, percentile_99)
                    Brier99.append((prob_AWAP_99 - prob_forecast_99) ** 2)
                    prob_AWAP_995 = calAWAPprob(b, percentile_995)
                    prob_forecast_995 = calforecastprob(a, percentile_995)
                    Brier995.append((prob_AWAP_995 - prob_forecast_995) ** 2)
                    #prob_AWAP_heavy30 = calAWAPprob(b, np.full(hr.shape, 30))
                    #prob_forecast_heavy30 = calforecastprob(a, np.full(hr.shape, 30))
                    #heavy30.append((prob_AWAP_heavy30 - prob_forecast_heavy30) ** 2)
                    #prob_AWAP_999 = calAWAPprob(b, percentile_999)
                    #prob_forecast_999 = calforecastprob(a, percentile_999)
                    #Brier999.append((prob_AWAP_999 - prob_forecast_999) ** 2)
                    mae_median_score = mae_median(a, b)
                    bias_relative_3 = bias_relative(a, b, constant=3)
                    bias_relative_5 = bias_relative_median(a, b, constant=5)
                    mean_bias_model.append(bias_QM)
                    mean_bias_median_model.append(bias_median_qm)
                    mae_score.append(QM_mae)

                    mean_crps_model.append(skill_QM)
                    alpha.append(QM_alpha)
                    mae_median_score_model.append(mae_median_score)
                    # mean_bias_relative_model_half.append(bias_relative_half)
                    # mean_bias_relative_model_1.append(bias_relative_1)
                    # mean_bias_relative_model_2.append(bias_relative_2)
                    # mean_bias_relative_model_2d9.append(bias_relative_2d9)
                    mean_bias_relative_model_3.append(bias_relative_3)
                    mean_bias_relative_model_5.append(bias_relative_5)

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
        folders = ["bias/QM", "bias_median/QM","mae/QM", "mae_median/QM","Brier0/QM", "Brier95/QM", "Brier99/QM", "Brier995/QM","heavy30/QM","crps_ss/QM", "relative_bias/QM","relative_bias_5/QM","alpha/QM"]

        # 循环创建所需的文件夹路径
        for folder in folders:
            folder_path = os.path.join(base_path, folder, str(year))
            os.makedirs(folder_path, exist_ok=True)  # 创建文件夹，如果已存在则忽略
            data_to_save = {
                "bias/QM": np.mean(mean_bias_model, axis=0),
                "bias_median/QM": np.mean(mean_bias_median_model, axis=0),
                "mae/QM": np.mean(mae_score, axis=0),
                "mae_median/QM": np.mean(mae_median_score_model, axis=0),
                "Brier0/QM": np.mean(Brier_0, axis=0),
                "Brier95/QM": np.mean(Brier95, axis=0),
                "Brier99/QM": np.mean(Brier99, axis=0),
                "Brier995/QM": np.mean(Brier995, axis=0),
                "heavy30/QM": np.mean(heavy30, axis=0),
                #"Brier999/QM": np.mean(Brier999, axis=0),
                "crps_ss/QM": np.mean(mean_crps_model, axis=0),
                "relative_bias/QM": np.mean(mean_bias_relative_model_3, axis=0),
                "relative_bias_5/QM": np.mean(mean_bias_relative_model_3, axis=0),
                "alpha/QM": np.mean(alpha, axis=0)
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
    