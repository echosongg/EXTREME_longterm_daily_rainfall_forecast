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
    Brier_startyear = 1976
    Brier_endyear = 2005
    percentile_95 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    percentile_99 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    percentile_995 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99.5)
    percentile_999 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99.9)
    dates_needs = dpt.date_range(date(year, 1, 1), date(year + 1, 7, 29))
    file_BARRA_dir = "/scratch/iu60/xs5813/Awap_data_bigger/"
    #     date_map=np.array(dates_needs)
    # np.where(date_map==date(2012, 1, 1))
    crps_ref = []
    log_crps_ref = []
    Brier_0 = []
    Brier95 = []
    Brier99 = []
    Brier995 = []
    Brier999 = []
    heavy30 = []
    mae_score = []
    bias_ref = []
    bias_median_ref = []
    mae_median_ref = []
    bias_relative_ref = []
    mean_bias_relative_model_half = []
    mean_bias_relative_model_1 = []
    mean_bias_relative_model_2 = []
    mean_bias_relative_model_2d9 = []
    mean_bias_relative_model_3 = []
    mean_bias_relative_model_5 = []
    alpha = []

    for target_date in dates_needs:
        hr, Awap_date = dpt.read_awap_data_fc(file_BARRA_dir, target_date)
        #hr_log, Awap_date_log = dpt.read_awap_data_fc_log(file_BARRA_dir, target_date)
        print('hr: ', hr)
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
            prob_awap0 = calAWAPdryprob(hr, 0.1)
            prob_awap95 = calAWAPprob(hr, percentile_95)
            #print("prob_awap95",prob_awap95)
            prob_awap99 = calAWAPprob(hr, percentile_99)
            #print("prob_awap99",prob_awap99)
            prob_awap995 = calAWAPprob(hr, percentile_995)
            prob_awap30 = calAWAPprob(hr, np.full(hr.shape, 30))

            #prob_awap999 = calAWAPprob(hr, percentile_999)
            #print("prob_awap999",prob_awap999)
            #are_equal = np.logical_and(prob_awap95 == prob_awap99, prob_awap99 == prob_awap999)
            #print("Are all probabilities equal?", are_equal.all())
            #Are all probabilities equal? True
            prob_ensamble0 = calforecastdryprob(ensamble, 0.1)
            prob_ensamble95 = calforecastprob(ensamble, percentile_95)
            prob_ensamble99 = calforecastprob(ensamble, percentile_99)
            prob_ensamble995 = calforecastprob(ensamble, percentile_995)
            prob_ensamble30 = calforecastprob(ensamble, np.full(hr.shape, 30))
            #prob_ensamble999 = calforecastprob(ensamble, percentile_999)
            climat_mae = mae(ensamble, hr)
            mae_median_score = mae_median(ensamble, hr)
            bias_score = bias(ensamble, hr)
            bias_median_score = bias_median(ensamble, hr)
            # bias_relative_half = bias_relative_median(ensamble, hr, constant=0.5)
            # bias_relative_1 = bias_relative_median(ensamble, hr, constant=1)
            # bias_relative_2 = bias_relative_median(ensamble, hr, constant=2)
            # bias_relative_2d9 = bias_relative_median(ensamble, hr, constant=2.9)
            bias_relative_3 = bias_relative(ensamble, hr, constant=3)
            bias_relative_5 = bias_relative_median(ensamble, hr, constant=5)
            clim_alpha = calculate_alpha_index(ensamble, hr)
            #
            Brier_0.append((prob_awap0 - prob_ensamble0) ** 2)
            Brier95.append((prob_awap95 - prob_ensamble95) ** 2)
            Brier99.append((prob_awap99 - prob_ensamble99) ** 2)
            Brier995.append((prob_awap995 - prob_ensamble995) ** 2)
            #heavy30.append((prob_awap30 - prob_ensamble30) ** 2)
            #Brier999.append((prob_awap999 - prob_ensamble999) ** 2)
            crps_ref.append(a)
            mae_score.append(climat_mae)
            mae_median_ref.append(mae_median_score)
            bias_ref.append(bias_score)
            bias_median_ref.append(bias_median_score)
            # mean_bias_relative_model_half.append(bias_relative_half)
            # mean_bias_relative_model_1.append(bias_relative_1)
            # mean_bias_relative_model_2.append(bias_relative_2)
            # mean_bias_relative_model_2d9.append(bias_relative_2d9)
            mean_bias_relative_model_3.append(bias_relative_3)
            mean_bias_relative_model_5.append(bias_relative_5)
            alpha.append(clim_alpha)
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
    # Your save operations
    np.save(os.path.join(base_path, f"climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(crps_ref, dtype=np.float64))
    #np.save(os.path.join(base_path, f"log_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(log_crps_ref, dtype=np.float64))
    np.save(os.path.join(base_path, f"prob0_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(Brier_0, dtype=np.float64))
    np.save(os.path.join(base_path, f"prob95_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(Brier95, dtype=np.float64))
    np.save(os.path.join(base_path, f"prob99_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(Brier99, dtype=np.float64))
    np.save(os.path.join(base_path, f"prob995_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(Brier995, dtype=np.float64))
    np.save(os.path.join(base_path, f"heavy30_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(heavy30, dtype=np.float64))
    #np.save(os.path.join(base_path, f"prob999_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(Brier999, dtype=np.float64))
    np.save(os.path.join(base_path, f"mae_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(mae_score, dtype=np.float64))
    np.save(os.path.join(base_path, f"mae_median_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(mae_median_ref, dtype=np.float64))
    np.save(os.path.join(base_path, f"relative_bias_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(mean_bias_relative_model_3, dtype=np.float64))
    np.save(os.path.join(base_path, f"relative_bias_5_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(mean_bias_relative_model_5, dtype=np.float64))
    np.save(os.path.join(base_path, f"bias_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(bias_ref, dtype=np.float64))
    np.save(os.path.join(base_path, f"bias_median_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(bias_median_ref, dtype=np.float64))
    np.save(os.path.join(base_path, f"alpha_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(alpha, dtype=np.float64))

if __name__ == '__main__':
    year_list = [2006,2007,2018]
    timewind = [1]
    for i in timewind:
        for j in year_list:
            main(j, i)
            print(str(j) + ", " + str(i * 2 - 1))
