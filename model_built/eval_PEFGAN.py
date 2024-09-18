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
import properscoring as ps

def calculate_alpha_index(ensemble_forecasts, observations):
    """
    Optimized calculation of the alpha index for forecast reliability based on 3D PIT values.

    Parameters:
    pit_values (array-like): A 3D array of PIT (Probability Integral Transform) values,
                             where the first two dimensions are spatial (x, y) and the third
                             dimension represents ensemble members.

    Returns:
    np.ndarray: A 2D array of alpha index values where each element corresponds
                to a spatial point (x, y) in the grid.
    """
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))
    pit_values = ensemble_forecasts <= observations[:, :, np.newaxis]
    # Get the shape of the PIT values array
    x_size, y_size, ensemble_size = pit_values.shape
    #print(f"Shape of pit_values: {pit_values.shape}") 

    # Sort the PIT values along the ensemble dimension (axis 2)
    pit_sorted = np.sort(pit_values, axis=2)

    # Calculate expected uniform distribution values for the ensemble members (broadcasted)
    expected_uniform = np.linspace(1 / (ensemble_size + 1), ensemble_size / (ensemble_size + 1), ensemble_size)
    # Calculate the absolute differences between sorted PIT values and the expected uniform distribution
    absolute_differences = np.abs(pit_sorted - expected_uniform[None, None, :])

    # Sum the absolute differences along the ensemble member dimension (axis 2)
    sum_absolute_differences = np.sum(absolute_differences, axis=2)

    # Calculate the alpha index for each spatial point
    alpha_values = 1 - (2 / ensemble_size) * sum_absolute_differences
    print(f"Shape of sum_absolute_differences: {sum_absolute_differences.shape}") 

    return alpha_values
    
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
    
# def relative_bias(ens, hr):

#     return (ens - hr).sum(axis=0) / ens.shape[0] / hr

def rmse(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt(((ens - hr) ** 2).sum(axis=(0)) / ens.shape[0])

def mae(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens - hr)).sum(axis=0) / ens.shape[0]

# ===========================================================
# Training settings
# ===========================================================


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
        print(self.file_AWAP_dir)
        print(self.file_ACCESS_dir)
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
        print(self.filename_list)
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
                    #                   print(access_path)
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
            self.file_ACCESS_dir, en, access_date, time_leading, self.year, "pr")
        #lr_log = dpt.read_access_data_calibrataion_log(
            #self.file_ACCESS_dir, en, access_date, time_leading, year, "pr")
        label, AWAP_date = dpt.read_awap_data_fc(self.file_AWAP_dir, awap_date)
        #label_log, AWAP_date_log = dpt.read_awap_data_fc_log(self.file_AWAP_dir, awap_date)

        return np.array(lr), np.array(label), torch.tensor(
            int(en[1:])), torch.tensor(int(access_date.strftime("%Y%m%d"))),  torch.tensor(int(AWAP_date.strftime("%Y%m%d"))), torch.tensor(time_leading)


def write_log(log, args):
    if not os.path.exists("./save/" + args.train_name + "/"):
        os.mkdir("./save/" + args.train_name + "/")
    my_log_file = open("./save/" + args.train_name + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


def main(year, days):

    model_name = 'model_G_i000007_20240910-042620'
    #model_name = 'model_G_i000008_20240824-212330_with_huber'
    version = "TestRefactored"
    #30 year
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

    write_log("start", args)
    percentile_95 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    print("percentile 95 is : ", percentile_95)
    # print("type of percentile95", type(percentile_95))
    # print("The size of percentile95 ", len(percentile_95))
    # print("The size of percentile95[0] ", len(percentile_95[0]))
    # print('Maximum  value of percentile 95 ', percentile_95.max())
    percentile_99 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    print("percentile 99 is : ", percentile_99)
    percentile_995 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99.5)

    def compute_metrics(sr, hr, args):
        print("hr.shape",hr.shape)
        print("sr.shape",np.transpose(sr, (1, 2, 0)).shape)
        metrics = {
            # "mae": mae(sr, hr),
            # "mae_mean": mae_mean(sr, hr),
            # "mae_median": mae_median(sr, hr),
            # "bias": bias(sr, hr),
            # "bias_median": bias_median(sr, hr),
            # "rmse": rmse(sr, hr),
            # "skil": ps.crps_ensemble(hr, np.transpose(sr, (1, 2, 0))),
            # "relative_bias_5": bias_relative(sr, hr, constant=5),
            # #"Brier_0": brier_score(calAWAPdryprob(hr, 0.1), calforecastdryprob(sr, 0.1)),
            # "Brier_95": brier_score(calAWAPprob(hr, percentile_95), calforecastprob(sr, percentile_95)),
            # "Brier_99": brier_score(calAWAPprob(hr, percentile_99), calforecastprob(sr, percentile_99)),
            # "Brier_995": brier_score(calAWAPprob(hr, percentile_995), calforecastprob(sr, percentile_995)),
            "alpha": calculate_alpha_index(sr, hr)
        }
        return metrics

    def brier_score(prob_AWAP, prob_forecast):
        return (prob_AWAP - prob_forecast) ** 2

    for lead in range(0, days):
        args.leading_time_we_use = lead
        print("lead", lead)

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms, hr_transform=hr_transforms, shuffle=False, args=args)
        print("data_set length:", len(data_set))
        test_data = DataLoader(data_set, batch_size=18, shuffle=False, num_workers=args.n_threads, drop_last=True)

        results = {metric: [] for metric in ["alpha"]}#"mae", "mae_mean", "mae_median", "bias", "bias_median", "rmse", "skil", "relative_bias_5", "Brier_95", "Brier_99","Brier_995", 
        for batch, (pr, hr, _, access_date, awap_date, _) in enumerate(test_data):
            with torch.no_grad():
                sr_np = pr.cpu().numpy()
                hr_np = hr.cpu().numpy()

                print("sr:", sr_np.shape)
                print("hr:", hr_np.shape)
                print("ACCESS_date", access_date)
                print("AWAP_date", awap_date)

                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b = np.squeeze(hr_np[i * args.ensemble])
                    metrics = compute_metrics(a, b, args)
                    # print("Values of a:", a)
                    # print("Shape of a:", a.shape)
                    # print("Max value of a:", np.max(a))
                    # print("Min value of a:", np.min(a))

                    # print("Values of b:", b)
                    # print("Shape of b:", b.shape)
                    # print("Max value of b:", np.max(b))
                    # print("Min value of b:", np.min(b))
                    for key, value in metrics.items():
                        results[key].append(value)

        
        base_path = "/scratch/iu60/xs5813/metric_results/"
        
        for key in results:
            # 计算每个度量的平均值
            mean_value = np.mean(results[key], axis=0)
            
            
            folder_path = f"{base_path}{key}/{model_name}/{year}/"
            print("folder_path:",folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            file_name = f"lead_time{lead}_whole.npy"
            
            # 保存计算得到的平均值到.npy文件
            np.save(os.path.join(folder_path, file_name), mean_value)
            print(f"save {key}")
                                
years = [2006, 2007, 2018]

if __name__ == '__main__':
    for year in years:
        main(year=year, days=42)
