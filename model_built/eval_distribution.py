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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calAWAPprob(AWAP_data, percentile):
    ''' 
    input: AWAP_data is  413 * 267
            percentile size is 413 * 267
    return: A probability matrix which size is 413 * 267 indicating the probability of the values in ensemble forecast 
    is greater than the value in the same pixel in percentile matrix

    '''

    
    return (AWAP_data > percentile) * 1

def calforecastprob_from_distribution(p_pred, alpha_pred, beta_pred, y_true, percentile, shave_border=0, num_values=10):
    # Initialize prediction matrices
    p_pred = torch.tensor(p_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.tensor(alpha_pred, dtype=torch.float32, device=device)
    beta_pred = torch.tensor(beta_pred, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
    alpha_pred = torch.clamp(alpha_pred, min=1e-6)
    beta_pred = torch.clamp(beta_pred, min=1e-6)
    
    if isinstance(percentile, (int, float)):
        percentile = torch.tensor(percentile, dtype=torch.float32, device=device)
    else:
        percentile = torch.tensor(percentile, dtype=torch.float32, device=device)

    forecasts = torch.zeros((num_values, *p_pred.shape), dtype=torch.float32, device=device)

    for i in range(num_values):
        is_rain = torch.bernoulli(p_pred)
        rain_amount = torch.distributions.gamma.Gamma(alpha_pred, beta_pred).sample()
        forecasts[i] = is_rain * rain_amount

    forecasts = torch.expm1(forecasts * 7)
    forecasts = forecasts.view(-1, *y_true.shape)

    if shave_border > 0:
        forecasts = forecasts[:, shave_border:-shave_border, shave_border:-shave_border]

    prob_matrix = (forecasts > percentile).float() 
    return torch.mean(prob_matrix, dim=0)

def calforecastprob(p_pred, alpha_pred, beta_pred, percentile):
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
    
def mae_median(p_pred, alpha_pred, beta_pred, hr, num_values=30):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    p_pred = torch.tensor(p_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.tensor(alpha_pred, dtype=torch.float32, device=device)
    beta_pred = torch.tensor(beta_pred, dtype=torch.float32, device=device)
    hr = torch.tensor(hr, dtype=torch.float32, device=device)
    alpha_pred = torch.clamp(alpha_pred, min=1e-6)
    beta_pred = torch.clamp(beta_pred, min=1e-6)
    print("p_pred",p_pred.shape)
    forecasts = torch.zeros((num_values, *p_pred.shape), dtype=torch.float32, device=device)
    for i in range(num_values):
        is_rain = torch.bernoulli(p_pred)
        rain_amount = torch.distributions.gamma.Gamma(alpha_pred, 1/beta_pred).sample()
        forecasts[i] = is_rain * rain_amount
    forecasts = torch.expm1(forecasts * 7)
    forecasts = forecasts.view(-1, *hr.shape)
    
    #forecasts = torch.minimum(forecasts, 1.1 * history) # Limit values to historical max
    median_forecasts = torch.median(forecasts, axis=0).values  # 修改这里，获取median的values属性
    return torch.abs(median_forecasts - hr)

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
    my_log_file = open("./save/" + args.train_name + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return

def CRPS_from_distribution(p_pred, alpha_pred, beta_pred, y_true, history, shave_border=0, num_values = 10):
    # Initialize prediction matrices
    p_pred = torch.tensor(p_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.tensor(alpha_pred, dtype=torch.float32, device=device)
    beta_pred = torch.tensor(beta_pred, dtype=torch.float32, device=device)
    alpha_pred = torch.clamp(alpha_pred, min=1e-6)
    beta_pred = torch.clamp(beta_pred, min=1e-6)
    y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
    history = torch.tensor(history, dtype=torch.float32, device=device)
    forecasts = torch.zeros((num_values, *p_pred.shape), dtype=torch.float32, device=device)
    # Generate 10 predicted values based on Gamma distribution
    for i in range(num_values):
        is_rain = torch.bernoulli(p_pred)
        rain_amount = torch.distributions.gamma.Gamma(alpha_pred, 1/beta_pred).sample()
        forecasts[i] = is_rain * rain_amount  # If no rain, rain amount is 0
    
    # Remove border pixels
    forecasts = torch.expm1(forecasts * 7)
    forecasts = forecasts.view(-1, *y_true.shape)
    # Calculate CRPS
    print("Shape of pred before squeeze:",p_pred.shape)
    print("Shape of y_true before squeeze:",y_true.shape)
    print("Shape of forecasts before squeeze:", forecasts.shape)
    crps = ps.crps_ensemble(y_true.cpu().numpy(), forecasts.cpu().numpy().transpose(1, 2, 0))
    crps = torch.tensor(crps, dtype=torch.float32, device=device)
    return crps

def main(year, days):

    model_name = 'model_G_i000005_20240403-035316_with_huber'
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

    write_log("start", args)
    percentile_95 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    print("percentile 95 is : ", percentile_95)
    # print("type of percentile95", type(percentile_95))
    # print("The size of percentile95 ", len(percentile_95))
    # print("The size of percentile95[0] ", len(percentile_95[0]))
    # print('Maximum  value of percentile 95 ', percentile_95.max())
    percentile_99 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    percentile_995 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99.5)
    #print("args.test_start_time",args.test_start_time)
    #print("args.access_path",args.file_ACCESS_dir)
    #test_instance = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms, hr_transform=hr_transforms, shuffle=False, args=args)
    #print(test_instance.__getitem__(0))  # 尝试获取第一个元素，看是否能正常工作
    def compute_metrics(sr, hr, args):
        metrics = {
            "skil_dis": CRPS_from_distribution(np.squeeze(sr[:, 0, :, :]), np.squeeze(sr[:, 1, :, :]), np.squeeze(sr[:, 2, :, :]), hr),
            #change
            #"Brier_0": brier_score(calAWAPdryprob(hr, 0.1), calforecastdryprob(sr, 0.1)),
            "Brier_0_dis": brier_score(calAWAPdryprob(hr, 0.1), np.squeeze(sr[:, 0, :, :])),
            #"Brier_95": brier_score(calAWAPprob(hr, percentile_95), calforecastprob(sr, percentile_95)),
            "Brier_95_dis": brier_score(calAWAPprob(hr, percentile_95), calforecastprob_from_distribution(np.squeeze(sr[:, 0, :, :]), np.squeeze(sr[:, 1, :, :]), np.squeeze(sr[:, 2, :, :]), hr, percentile_95)),
            #"Brier_99": brier_score(calAWAPprob(hr, percentile_99), calforecastprob(sr, percentile_99)),
            "Brier_99_dis": brier_score(calAWAPprob(hr, percentile_99), calforecastprob_from_distribution(np.squeeze(sr[:, 0, :, :]), np.squeeze(sr[:, 1, :, :]), np.squeeze(sr[:, 2, :, :]), hr, percentile_99)),
            "Brier_995_dis": brier_score(calAWAPprob(hr, percentile_995), calforecastprob_from_distribution(np.squeeze(sr[:, 0, :, :]), np.squeeze(sr[:, 1, :, :]), np.squeeze(sr[:, 2, :, :]), hr, percentile_995)),
        }
        return metrics
    def brier_score(prob_AWAP, prob_forecast): 
        prob_AWAP = torch.tensor(prob_AWAP, dtype=torch.float32, device=device)
        prob_forecast = torch.tensor(prob_forecast, dtype=torch.float32, device=device)
        metric_data = (prob_AWAP - prob_forecast) ** 2
        return metric_data

    for lead in range(0, days):
        args.leading_time_we_use = lead
        print("lead", lead)

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms, hr_transform=hr_transforms, shuffle=False, args=args)
        print("data_set length:", len(data_set))
        test_data = DataLoader(data_set, batch_size=18, shuffle=False, num_workers=args.n_threads, drop_last=True)

        results = {metric: [] for metric in ["skil_dis","Brier_0_dis", "Brier_95_dis", "Brier_99_dis","Brier_995_dis"]}

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
                results[key] = torch.stack(results[key]).cpu().numpy()
                mean_value = np.mean(results[key], axis=0)
                print(f"Average of {key}: {np.mean(mean_value)}")

                folder_path = f"{base_path}{key}/{model_name}/{year}/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                file_name = f"lead_time{lead}_whole.npy"
                np.save(os.path.join(folder_path, file_name), mean_value)
            else:
                print(f"No results for {key}")
        
if __name__ == '__main__':
    years = [2006, 2007, 2018]
    days = 42  # Assuming days remain constant for each year.
    for year in years:
        main(year, days)
        print(f'EXTRME {year} done')