import data_processing_tool as dpt
import random
from torch.utils.data import Dataset
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import platform
from datetime import timedelta, date, datetime
import numpy as np
import os
import time
import logging
import properscoring as ps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建./save目录，如果不存在的话
log_dir = './save'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 定义日志文件路径
log_file = os.path.join(log_dir, 'disttibution.log')

# 配置logging模块
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志记录级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志输出格式
    handlers=[
        logging.FileHandler(log_file),  # 将日志输出到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

#def calculate_pit_values(ensemble_forecasts, observations, epsilon=1e-6):
def calculate_pit_values(p_pred, alpha_pred, beta_pred, observations, history, shave_border=0, num_values=10, epsilon = 1e-6):
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
    p_pred = torch.tensor(p_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.tensor(alpha_pred, dtype=torch.float32, device=device)
    beta_pred = torch.tensor(beta_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.clamp(alpha_pred, min=1e-6)
    beta_pred = torch.clamp(beta_pred, min=1e-6)
    observations = torch.tensor(observations, dtype=torch.float32, device=device)
    history = torch.tensor(history, dtype=torch.float32, device=device)
    ensemble_forecasts = torch.zeros((num_values, *p_pred.shape), dtype=torch.float32, device=device)
    # Generate 10 predicted values based on Gamma distribution
    for i in range(num_values):
        is_rain = torch.bernoulli(p_pred)
        rain_amount = torch.distributions.gamma.Gamma(alpha_pred, 1/beta_pred).sample()
        ensemble_forecasts[i] = is_rain * rain_amount  # If no rain, rain amount is 0
    
    # Remove border pixels
    ensemble_forecasts = torch.expm1(ensemble_forecasts * 7)
    ensemble_forecasts = ensemble_forecasts.view(-1, *observations.shape)
    ensemble_forecasts = torch.minimum(ensemble_forecasts, 1.1 * history) # Limit values to historical max
    ensemble_forecasts = ensemble_forecasts.cpu().numpy()
    observations = observations.cpu().numpy()
    print("ensemble_forecasts", ensemble_forecasts.shape)
    print("observations", observations.shape)


    # Check input shapes
    if ensemble_forecasts.ndim != 3 or observations.ndim != 2:
        raise ValueError(
            "Input dimensions are incorrect. ensemble_forecasts should be 3D and observations should be 2D.")
    #ensemble_forecasts (90, 413, 267)
    #observations (413, 267)
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))
    if ensemble_forecasts.shape[:2] != observations.shape:
        raise ValueError("Horizontal and vertical dimensions of ensemble_forecasts and observations must match.")

    # Count the total number of ensemble members
    n_ensemble = ensemble_forecasts.shape[-1]
    print("n_ensenmble: ", n_ensemble)

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
    print("pit value shape:",pit_values.shape)

    return pit_values

#def calculate_pit_values(ensemble_forecasts, observations, epsilon=1e-6):
def calculate_pit_values_plus(p_pred, alpha_pred, beta_pred, observations, history, shave_border=0, num_values=10, epsilon = 1e-6):
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
    p_pred = torch.tensor(p_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.tensor(alpha_pred, dtype=torch.float32, device=device)
    beta_pred = torch.tensor(beta_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.clamp(alpha_pred, min=1e-6)
    beta_pred = torch.clamp(beta_pred, min=1e-6)
    observations = torch.tensor(observations, dtype=torch.float32, device=device)
    history = torch.tensor(history, dtype=torch.float32, device=device)
    ensemble_forecasts = torch.zeros((num_values, *p_pred.shape), dtype=torch.float32, device=device)
    # Generate 10 predicted values based on Gamma distribution
    for i in range(num_values):
        is_rain = torch.bernoulli(p_pred)
        rain_amount = torch.distributions.gamma.Gamma(alpha_pred, 1/beta_pred).sample()
        ensemble_forecasts[i] = is_rain * rain_amount  # If no rain, rain amount is 0
    
    # Remove border pixels
    ensemble_forecasts = torch.expm1(ensemble_forecasts * 7)
    ensemble_forecasts = ensemble_forecasts.view(-1, *observations.shape)
    ensemble_forecasts = torch.minimum(ensemble_forecasts, 1.1 * history) # Limit values to historical max
    ensemble_forecasts = ensemble_forecasts.cpu().numpy()
    observations = observations.cpu().numpy()
    print("ensemble_forecasts", ensemble_forecasts.shape)
    print("observations", observations.shape)


    # Check input shapes
    if ensemble_forecasts.ndim != 3 or observations.ndim != 2:
        raise ValueError(
            "Input dimensions are incorrect. ensemble_forecasts should be 3D and observations should be 2D.")
    #ensemble_forecasts (90, 413, 267)
    #observations (413, 267)
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))
    if ensemble_forecasts.shape[:2] != observations.shape:
        raise ValueError("Horizontal and vertical dimensions of ensemble_forecasts and observations must match.")

    # Count the total number of ensemble members
    n_ensemble = ensemble_forecasts.shape[-1]+1
    print("n_ensenmble: ", n_ensemble)

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
    print("pit value shape:",pit_values.shape)

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

    print("alpha_values",alpha_values.shape)

    return alpha_values



class ACCESS_AWAP_cali(Dataset):
    '''
2.using my net to train one channel to one channel.
    '''

    def __init__(self, start_date=date(2007, 1, 1), end_date=date(2007, 12, 31), regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, args=None):
        #         print("=> BARRA_R & ACCESS_S1 loading")
        #         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        # '/g/data/rr8/OBS/AWAP_ongoing/v0.6/grid_05/daily/precip/'
        self.file_AWAP_dir = args.file_AWAP_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args = args

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date

        self.regin = regin
        self.leading_time_we_use = args.leading_time_we_use

        self.ensemble_access = ['e01', 'e02', 'e03', 'e04',
                                'e05', 'e06', 'e07', 'e08', 'e09']
        self.ensemble = []
        for i in range(len(self.ensemble_access)):
            self.ensemble.append(self.ensemble_access[i])

        self.dates = self.date_range(start_date, end_date)

        self.filename_list, self.year = self.get_filename_with_time_order(
            self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir)
            print("no file or no permission")
        #en, cali_date, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        #data_awap = dpt.read_awap_data_fc_get_lat_lon(
        #    self.file_AWAP_dir, date_for_AWAP)
        #self.lat_awap = data_awap[1]
        #self.lon_awap = data_awap[2]

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

    def get_filename_with_time_order(self, rootdir):#问题
        '''get filename first and generate label ,one different w'''
        _files = []
        year = 0
        for date in self.dates:
            for i in range(self.leading_time_we_use, self.leading_time_we_use + 1):

                for en in self.ensemble:
                    access_path = rootdir + en + "/pr/" + date.strftime("%Y") + "/" + date.strftime("%Y-%m-%d") + ".nc"
                    #access_path = rootdir + en + "/pr/"  + date.strftime("%Y-%m-%d") + ".nc"
                    if os.path.exists(access_path):

                        if date == self.end_date and i == 1:
                            break
                        path = []
                        path.append(en)
                        awap_date = date + timedelta(i)
                        path.append(date)
                        path.append(awap_date)
                        path.append(i)
                        _files.append(path)
                        year = date.strftime("%Y")

        # 最后去掉第一行，然后shuffle
        #en, access_date, awap_date, time_leading
        return _files, year

    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t = time.time()
        print("getitem")

        # read_data filemame[idx]
        #print("self.filename_list",self.filename_list)
        en, access_date, awap_date, time_leading = self.filename_list[idx]
        lr = dpt.read_access_data_calibration(
            self.file_ACCESS_dir, en, access_date, time_leading, self.year, ["p","alpha", "beta"])
        #lr_log = dpt.read_access_data_calibrataion_log(
            #self.file_ACCESS_dir, en, access_date, time_leading, year, "pr")
        label, AWAP_date = dpt.read_awap_data_fc(self.file_AWAP_dir, awap_date)
        #label_log, AWAP_date_log = dpt.read_awap_data_fc_log(self.file_AWAP_dir, awap_date)

        return np.array(lr), np.array(label), torch.tensor(
            int(en[1:])), torch.tensor(int(access_date.strftime("%Y%m%d"))),  torch.tensor(int(AWAP_date.strftime("%Y%m%d"))), torch.tensor(time_leading)


def write_log(log, args):
    print(log)
    if not os.path.exists("./save/" + args.train_name + "/"):
        os.mkdir("./save/" + args.train_name + "/")
    my_log_file = open("./save/" + args.train_name + '/distribution.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return

def main(year, days):

    model_name = 'model_G_i000008_20240824-212330_with_huber'
    version = "TestRefactored"
    Brier_startyear = 1976
    Brier_endyear = 2005
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true', help='cpu only?')

    # hyper-parameters
    parser.add_argument('--train_name', type=str,
                        default="cali_crps", help='training name')

    parser.add_argument('--batch_size', type=int,
                        default=18, help='training batch size')
    parser.add_argument('--testBatchSize', type=int,
                        default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=200,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed to use. Default=123')

    # model configuration
    parser.add_argument('--upscale_factor', '-uf', type=int,
                        default=4, help="super resolution upscale factor")
    parser.add_argument('--model', '-m', type=str, default='vdsr',
                        help='choose which model is going to use')

    # data
    parser.add_argument('--pr', type=bool, default=True, help='add-on pr?')

    parser.add_argument('--train_start_time', type=type(datetime(1990,
                                                                 1, 25)), default=datetime(1990, 1, 2), help='r?')
    parser.add_argument('--train_end_time', type=type(datetime(1990,
                                                               1, 25)), default=datetime(1990, 2, 9), help='?')
    parser.add_argument('--test_start_time', type=type(datetime(2012,
                                                                1, 1)), default=datetime(2012, 1, 1), help='a?')
    parser.add_argument('--test_end_time', type=type(datetime(2012,
                                                              12, 31)), default=datetime(2012, 12, 31), help='')

    parser.add_argument('--dem', action='store_true', help='add-on dem?')
    parser.add_argument('--psl', action='store_true', help='add-on psl?')
    parser.add_argument('--zg', action='store_true', help='add-on zg?')
    parser.add_argument('--tasmax', action='store_true', help='add-on tasmax?')
    parser.add_argument('--tasmin', action='store_true', help='add-on tasmin?')
    parser.add_argument('--leading_time_we_use', type=int,
                        default=1, help='add-on tasmin?')
    parser.add_argument('--ensemble', type=int, default=9,
                        help='total ensambles is 9')
    parser.add_argument('--channels', type=float, default=0,
                        help='channel of data_input must')
    # [111.85, 155.875, -44.35, -9.975]
    parser.add_argument('--domain', type=list,
                        default=[140.6, 153.9, -39.2, -18.6], help='dataset directory')
    
    parser.add_argument('--file_ACCESS_dir', type=str,
                        #default="/scratch/iu60/xs5813/TestResults/"+ model_name + '/' + str(year) + "/" + model_num + "/",
                        default="/scratch/iu60/xs5813/TestResults/DESRGAN/v" + str(version) + "/"  + str(model_name) + "/",
                        help='dataset directory')
    parser.add_argument('--file_AWAP_dir', type=str, default="/scratch/iu60/xs5813/Awap_data_bigger/",
                        help='dataset directory')

    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half', 'double'),
                        help='FP precision for test (single | half)')

    args = parser.parse_args()
    sys = platform.system()
    args.dem = False
    args.train_name = "pr_DESRGAN"
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
    print("  evaluating model | %s" % args.file_ACCESS_dir[-17:])
    print("  ------------------------------")
    print("  num of channels | %5d" % args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d" % args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d" % args.batch_size)
    print("  ------------------------------")
    print("  using cpu only | %5d" % args.cpu)
    print("  ------------------------------")
    print("  The percentile year begin from | %5d" % int(Brier_startyear))
    print("  ------------------------------")
    print("  The percentile year end at | %5d" % int(Brier_endyear))

    lr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    hr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    args.test_start_time = datetime(year, 1, 1)
    args.test_end_time = datetime(year, 12, 31)
    history = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 100)
    def compute_metrics(sr, hr, args):
        metrics = {
            "alpha_dis_plus": calculate_pit_values_plus(np.squeeze(sr[:, 0, :, :]), np.squeeze(sr[:, 1, :, :]), np.squeeze(sr[:, 2, :, :]), hr, history)
        }
        return metrics

    for lead in range(0, days):
        args.leading_time_we_use = lead
        print("lead", lead)

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms, hr_transform=hr_transforms, shuffle=False, args=args)
        print("data_set length:", len(data_set))
        test_data = DataLoader(data_set, batch_size=18, shuffle=False, num_workers=args.n_threads, drop_last=True)

        results = {metric: [] for metric in ["alpha_dis_plus"]} #"skil_dis", "mae_median_dis","Brier_95_dis", "Brier_99_dis","Brier_995_dis"，

        for batch, (pr, hr, _, access_date, awap_date, _) in enumerate(test_data):
            with torch.no_grad():
                sr_np = pr.cpu().numpy()
                print("sr_np shape",sr_np.shape)
                hr_np = hr.cpu().numpy()

                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b = np.squeeze(hr_np[i * args.ensemble])
                    metrics = compute_metrics(a, b, args)
                    print("hr", hr.shape)

                    for key, value in metrics.items():
                        results[key].append(value)
        
        base_path = "/scratch/iu60/xs5813/metric_results/"
        
        for key in results:
            if results[key]:  # 确保列表非空
                results[key] = np.stack(results[key], axis=0)
                results[key] = calculate_alpha_index(results[key])
                mean_value = results[key]
                #print(f"Average of {key}: {np.mean(mean_value)}")

                folder_path = f"{base_path}{key}/{model_name}/{year}/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                file_name = f"lead_time{lead}_whole.npy"
                np.save(os.path.join(folder_path, file_name), mean_value)
            else:
                print(f"No results for {key}")
        
if __name__ == '__main__':
    years = [2006, 2018]
    days = 42  # Assuming days remain constant for each year.
    for year in years:
        main(year, days)
        print(f'EXTRME {year} done')