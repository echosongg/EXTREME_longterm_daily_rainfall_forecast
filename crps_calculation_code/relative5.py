# 计算climatology for validation for nci

# from scipy.stats import norm
# ps.crps_ensemble(obs,ens).shape
from os import mkdir
import os
from datetime import timedelta, date, datetime
import properscoring as ps
import numpy as np
import data_processing_tool as dpt
import sys

sys.path.append('../')


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

def main(year, time_windows):
    #30year
    dates_needs = dpt.date_range(date(year, 1, 1), date(year + 1, 7, 29))
    file_BARRA_dir = "/scratch/iu60/xs5813/Awap_data_bigger/"
    mean_bias_relative_model_5 = []

    for target_date in dates_needs:
        hr, Awap_date = dpt.read_awap_data_fc(file_BARRA_dir, target_date)
        #hr_log, Awap_date_log = dpt.read_awap_data_fc_log(file_BARRA_dir, target_date)
        ensamble = []
        log_ensamble = []
        #         for y in range(1990,target_date.year):
        #for y in range(1981, 2006):
        for y in range(year-30, year):

            if target_date.year == y:
                continue

            for w in range(1, time_windows):  # for what time of windows

                filename = file_BARRA_dir + \
                           str(y) + (target_date - timedelta(w)).strftime("-%m-%d") + ".nc"
                if os.path.exists(filename):
                    t = date(y, (target_date - timedelta(w)).month,
                             (target_date - timedelta(w)).day)
                    sr = dpt.read_awap_data_fc(file_BARRA_dir, t)
                    #sr_log = dpt.read_awap_data_fc_log(file_BARRA_dir, t)
                    ensamble.append(sr)
                    log_ensamble.append(sr_log)

                filename = file_BARRA_dir + \
                           str(y) + (target_date + timedelta(w)).strftime("-%m-%d") + ".nc"
                if os.path.exists(filename):
                    t = date(y, (target_date + timedelta(w)).month,
                             (target_date + timedelta(w)).day)

                    sr = dpt.read_awap_data_fc(file_BARRA_dir, t)
                    #sr_log = dpt.read_awap_data_fc_log(file_BARRA_dir, t)
                    ensamble.append(sr)
                    #log_ensamble.append(sr_log)

            filename = file_BARRA_dir + str(y) + target_date.strftime("-%m-%d") + ".nc"
            if os.path.exists(filename):
                t = date(y, target_date.month, target_date.day)
                print(t)
                sr, temp_date = dpt.read_awap_data_fc(file_BARRA_dir, t)
                #sr_log, temp_date_log = dpt.read_awap_data_fc_log(file_BARRA_dir, t)
                print('sr: ', sr)
                ensamble.append(sr)
                #log_ensamble.append(sr_log)
        if ensamble:
            print("calculate ensemble")
            ensamble = np.array(ensamble)
            print("ensemble.shape:", ensamble.shape)
            print("hr.shape:", hr.shape)
            print("date:", Awap_date)
            a = ps.crps_ensemble(hr, ensamble.transpose(1, 2, 0))
            bias_relative_5 = bias_relative_median(ensamble, hr, constant=5)
            mean_bias_relative_model_5.append(bias_relative_5)
        # if log_ensamble:
        #     print("calculate ensemble")
        #     log_ensamble = np.array(log_ensamble)
        #     print("log ensemble.shape:", log_ensamble.shape)
        #     print("hr.shape:", hr_log.shape)
        #     print("date:", Awap_date_log)
        #     a_log = ps.crps_ensemble(hr_log, log_ensamble.transpose(1, 2, 0))
        #     log_crps_ref.append(a_log)
    # CRPS-SS
    # if not os.path.exists('./save/crps_ss/climatology/'+str(year)):
    #     mkdir('./save/crps_ss/climatology/'+str(year))

    base_path = "/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology"

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)
    np.save(os.path.join(base_path, f"relative_bias_5_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(mean_bias_relative_model_5, dtype=np.float64))

if __name__ == '__main__':
    year_list = [2018]
    timewind = [1]
    for i in timewind:
        for j in year_list:
            main(j, i)
            print(str(j) + ", " + str(i * 2 - 1))
