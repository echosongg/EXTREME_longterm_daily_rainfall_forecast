import data_processing_tool as dpt
from percentilecal import AWAPcalpercentile
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


def mae(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens - hr)).sum(axis=0) / ens.shape[0]


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


# ===========================================================
# Training settings
# ===========================================================
def read_awap_data_fc_get_lat_lon(root_dir, date_time):  # precip_calib_0.05_1911
    # filename=root_dir+(date_time+timedelta(1)).strftime("%Y%m%d")+".nc"
    filename = root_dir + (date_time).strftime("%Y-%m-%d") + ".nc"
    data = Dataset(filename, 'r')
    lats = data['lat'][:]
    lons = data['lon'][:]
    var = data["precip"][:]
    var = var.filled(fill_value=0)
    var = np.squeeze(var)
    data.close()
    return var, lats, lons
    
def date_range(self, start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

class ACCESS_AWAP_cali(Dataset):
    '''
2.using my net to train one channel to one channel.
    '''

    def __init__(self, start_date=date(1990, 1, 1), end_date=date(1990, 12, 31), regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, args=None, summer=True):
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
        self.ensemble = ['e01', 'e02', 'e03']
        for i in range(len(self.ensemble_access)):
            self.ensemble.append(self.ensemble_access[i])
       if summer:  # 如果summer为True，则过滤夏季月份
            self.dates = [date for date in self.dates if date.month in [10, 11, 12, 1, 2, 3]]

        self.dates = self.date_range(start_date, end_date)

        self.filename_list = self.get_filename_with_time_order(
            self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir)
            print("no file or no permission")
        print(self.filename_list)
        en, cali_date, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        data_awap = read_awap_data_fc_get_lat_lon(
            self.file_AWAP_dir, date_for_AWAP)
        self.lat_awap = data_awap[1]
        self.lon_awap = data_awap[2]

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
                    for lead_time in range(self.leading_time_we_use, self.leading_time_we_use + 1): # 1 - leading_time_we_use
                        
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

        log_lr = read_access_data(self.file_ACCESS_dir, en, access_date, time_leading)
        log_hr = read_awap_data(self.file_AWAP_dir, awap_date)
        lr = np.expm1(log_lr * 7) 
        hr = np.expm1(log_hr * 7) 

        #utils lr shape (1, 21, 19)
        #utils hr shape (1, 151, 226)

        return lr, hr, log_lr, log_hr, access_date.strftime("%Y%m%d"), awap_date.strftime("%Y%m%d"), time_leading



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


def write_log(log, args):
    print(log)
    if not os.path.exists("./save/" + args.train_name + "/"):
        os.mkdir("./save/" + args.train_name + "/")
    my_log_file = open("./save/" + args.train_name + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true', help='cpu only?')

    # hyper-parameters
    parser.add_argument('--train_name', type=str,
                        default="cali_crps", help='training name')

    parser.add_argument('--batch_size', type=int,
                        default=9, help='training batch size')
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
                        default=[142, 152.3, -31.95, -23.4], help='dataset directory')
    #就是ACCESS_data
    parser.add_argument('--file_ACCESS_dir', type=str,
                        default="/scratch/iu60/xs5813/Test_on_year"+ model_name + '/' + str(year) + "/" + model_num + "/",
                        help='dataset directory')
    parser.add_argument('--file_AWAP_dir', type=str, default="/scratch/iu60/xs5813/Awap_pre_data/",
                        help='dataset directory')

    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half', 'double'),
                        help='FP precision for test (single | half)')

    return parser.parse_args()

def main(year, days):
#need change crps part, and other part need to be changed as well
    model_name = 'EXTREME'
    model_num = 'here_got_it_from_my_output'
    Brier_startyear = 1981
    Brier_endyear = 2010
    
    args = parse_args()

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
    percentile_95 = AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    print("percentile 95 is : ", percentile_95)
    # print("type of percentile95", type(percentile_95))
    # print("The size of percentile95 ", len(percentile_95))
    # print("The size of percentile95[0] ", len(percentile_95[0]))
    # print('Maximum  value of percentile 95 ', percentile_95.max())
    percentile_99 = AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    for lead in range(0, days):
        args.leading_time_we_use = lead

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms,
                                    hr_transform=hr_transforms, shuffle=False, args=args)
        print("data_set length:", data_set.__len__())
        test_data = DataLoader(data_set,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=args.n_threads, drop_last=False)
        
        Brier_0_model = []
        Brier_95_model = []
        Brier_99_model = []
        mean_mae_model = []
        mean_mae_mean_model = []
        mean_mae_mediam_model = []
        mean_bias_model = []
        mean_relative_bias_model = []
        mean_rmse_model = []
        mean_crps_model = []
        mean_crps_model_log = []
        mean_bias_median_model = []
        print("test_data length:", len(test_data))
        for batch, (pr, hr, pr_log, hr_log, access_date, awap_date, _) in enumerate(test_data):
            pr_log = 
            with torch.set_grad_enabled(False):

                sr_np = pr.cpu().numpy()
                hr_np = hr.cpu().numpy()
                sr_log_np = pr_log.cpu().numpy()
                hr_log_np = hr_log.cpu().numpy()
                print("sr:", sr_np.shape)
                print("hr:", hr_np.shape)
                # print("sr:", sr_log_np.shape)
                # print("hr:", hr_log_np.shape)
                print("ACCESS_date", access_date)
                print("AWAP_date", awap_date)
                # print("num of i", args.batch_size // args.ensemble)
                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(
                        sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    print("shape of a ", a.shape)
                    b = np.squeeze(hr_np[i * args.ensemble])
                    print("shape of b ", b.shape)
                    a_log = np.squeeze(
                        sr_log_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b_log = np.squeeze(hr_log_np[i * args.ensemble])
                    mae_DESRGAN = mae(a, b)
                    mae_mean_DESRGAN = mae_mean(a, b)
                    mae_median_DESRGAN = mae_median(a, b)
                    bias_DESRGAN = bias(a, b)
                    bias_median_DESRGAN = bias_median(a, b)
                    rmse_DESRGAN = rmse(a, b)
                    skil_DESRGAN = ps.crps_ensemble(b, np.transpose(a, (1, 2, 0)))
                    print("The shape of CRPS: ", skil_DESRGAN.shape)
                    skil_DESRGAN_log = ps.crps_ensemble(b_log, np.transpose(a_log, (1, 2, 0)))
                    relative_bias_DESRGAN = bias_relative(a, b, constant=3)
                    #calculating Brier score
                    prob_AWAP_0 = calAWAPdryprob(b, 0.1)
                    prob_forecast_0 = calforecastdryprob(a, 0.1)
                    Brier_0_model.append((prob_AWAP_0 - prob_forecast_0) ** 2)
                    prob_AWAP_95 = calAWAPprob(b, percentile_95)
                    prob_forecast_95 = calforecastprob(a, percentile_95)
                    print("prob of AWAP: ", prob_AWAP_95)
                    print("prob of Forecast: ", prob_forecast_95)
                    Brier_95_model.append((prob_AWAP_95 - prob_forecast_95) ** 2)
                    prob_AWAP_99 = calAWAPprob(b, percentile_99)
                    prob_forecast_99 = calforecastprob(a, percentile_99)
                    Brier_99_model.append((prob_AWAP_99 - prob_forecast_99) ** 2)


                    mean_mae_model.append(mae_DESRGAN)
                    mean_mae_mean_model.append(mae_mean_DESRGAN)
                    mean_mae_mediam_model.append(mae_median_DESRGAN)
                    mean_bias_model.append(bias_DESRGAN)
                    mean_bias_median_model.append(bias_median_DESRGAN)
                    mean_relative_bias_model.append(relative_bias_DESRGAN)
                    mean_rmse_model.append(rmse_DESRGAN)
                    mean_crps_model.append(skil_DESRGAN)
                    mean_crps_model_log.append(skil_DESRGAN_log)
        # print("lead time = ", lead)
        base_path = "scratch/iu60/xs5813/metric_results/"
        
        print("The length of mean_crps_model: ", len(mean_crps_model))
        print("The shape of the mean of mean_crps_model: ", np.mean(mean_crps_model, axis=0).shape)
        if not os.path.exists(f"{base_path}Brier0/" + model_name + "/"):
            os.mkdir(f"{base_path}Brier0/"+ model_name)
        if not os.path.exists(f"{base_path}Brier0/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}Brier0/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}Brier0/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}Brier0/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save(f"{base_path}Brier0/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(Brier_0_model, axis=0))

        if not os.path.exists(f"{base_path}Brier95/" + model_name + "/"):
            os.mkdir(f"{base_path}Brier95/"+ model_name)
        if not os.path.exists(f"{base_path}Brier95/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}Brier95/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}Brier95/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}Brier95/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save(f"{base_path}Brier95/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(Brier_95_model, axis=0))
        
        if not os.path.exists(f"{base_path}Brier99/" + model_name + "/"):
            os.mkdir(f"{base_path}Brier99/"+ model_name)
        if not os.path.exists(f"{base_path}Brier99/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}Brier99/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}Brier99/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}Brier99/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save(f"{base_path}Brier99/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(Brier_99_model, axis=0))

        if not os.path.exists(f"{base_path}mae/" + model_name + "/"):
            os.mkdir(f"{base_path}mae/"+ model_name)
        if not os.path.exists(f"{base_path}mae/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}mae/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}mae/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}mae/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save(f"{base_path}mae/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(mean_mae_model, axis=0))

        if not os.path.exists(f"{base_path}mae_mean/" + model_name + "/"):
            os.mkdir(f"{base_path}mae_mean/"+ model_name)
        if not os.path.exists(f"{base_path}mae_mean/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}mae_mean/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}mae_mean/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}mae_mean/" + model_name + "/" + str(year) + "/" + model_num)
        np.save(
            f"{base_path}mae_mean/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_mae_mean_model, axis=0))

        if not os.path.exists(f"{base_path}mae_median/" + model_name + "/"):
            os.mkdir(f"{base_path}mae_median/"+ model_name)
        if not os.path.exists(f"{base_path}mae_median/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}mae_median/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}mae_median/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}mae_median/" + model_name + "/" + str(year) + "/" + model_num)
        np.save(
            f"{base_path}mae_median/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_mae_mediam_model, axis=0))

        if not os.path.exists("/scratch/iu60/xs5813/v/new_crps/save/bias/" + model_name + "/"):
            os.mkdir(f"{base_path}bias/"+ model_name)
        if not os.path.exists(f"{base_path}bias/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}bias/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}bias/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}bias/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            f"{base_path}bias/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_bias_model, axis=0))

        if not os.path.exists(f"{base_path}bias_median/" + model_name + "/"):
            os.mkdir(f"{base_path}bias_median/"+ model_name)
        if not os.path.exists(f"{base_path}bias_median/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}bias_median/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}bias_median/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}bias_median/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            f"{base_path}bias_median/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_bias_median_model, axis=0))
        
        if not os.path.exists(f"{base_path}relative_bias/" + model_name + "/"):
            os.mkdir(f"{base_path}relative_bias/"+ model_name)
        if not os.path.exists(f"{base_path}relative_bias/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}relative_bias/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}relative_bias/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}relative_bias/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            f"{base_path}relative_bias/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_relative_bias_model, axis=0))

        if not os.path.exists(f"{base_path}rmse/" + model_name + "/"):
            os.mkdir(f"{base_path}rmse/"+ model_name)
        if not os.path.exists(f"{base_path}rmse/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}rmse/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}rmse/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}rmse/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            f"{base_path}rmse/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_rmse_model, axis=0))

        if not os.path.exists(f"{base_path}crps_ss/" + model_name + "/"):
            os.mkdir(f"{base_path}crps_ss/"+ model_name)
        if not os.path.exists(f"{base_path}crps_ss/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}crps_ss/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}crps_ss/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}crps_ss/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            f"{base_path}crps_ss/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_crps_model, axis=0))
        
        if not os.path.exists(f"{base_path}crps_ss_log/" + model_name + "/"):
            os.mkdir(f"{base_path}crps_ss_log/"+ model_name)
        if not os.path.exists(f"{base_path}crps_ss_log/" + model_name + "/" + str(year)):
            os.mkdir(f"{base_path}crps_ss_log/"+ model_name + "/" + str(year))
        if not os.path.exists(f"{base_path}crps_ss_log/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir(f"{base_path}crps_ss_log/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            f"crps_ss_log/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_crps_model_log, axis=0))

        


if __name__ == '__main__':
    main(year=2006, days=42)
    print('2006 done')

    # main(year=2012, days=42)
    # print('2012 done')