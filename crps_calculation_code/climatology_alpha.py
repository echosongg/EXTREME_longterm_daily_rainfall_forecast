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
year = 2018
window = 1
# norm = "bias_relative"  # ["crps_ss", "mae_median", "bias", "bias_median"]
norm = "climatology"  # ["crps_ss", "mae_median", "bias", "bias_median"]
# ===========================================================
# Training settings
# ===========================================================
class aaa(object):
    def __init__(self, lead):
        self.ensemble_access = ['e01', 'e02', 'e03', 'e04',
                                'e05', 'e06', 'e07', 'e08', 'e09']
        self.lead_time = lead 
        self.rootdir = "/scratch/iu60/xs5813/Processed_data_bigger/"
        self.files = self.get_filename_with_time_order(self.rootdir)
        print("Including the files: ", self.files)
    def get_filename_with_time_order(self, rootdir):
        _files = []
        # Initial dates for ACCESS-S2
        date_dict = {1: [1, 14, 15, 16, 30, 31], 2: [1, 14, 15, 16, 27, 28], 3: [1, 14, 15, 16, 30, 31],
                     4: [1, 14, 15, 16, 29, 30], 5: [1, 14, 15, 16, 30, 31], 6: [1, 14, 15, 16, 29, 30],
                     7: [1, 14, 15, 16, 30, 31], 8: [1, 14, 15, 16, 30, 31], 9: [1, 14, 15, 16, 29, 30],
                     10: [1, 14, 15, 16, 30, 31], 11: [1, 14, 15, 16, 29, 30], 12: [1, 14, 15, 16, 30, 31]}
        for mm in range(1, 13):
            for dd in date_dict[mm]:
                date_time = date(year, mm, dd)
                access_path = rootdir + "e09/da_pr_" + date_time.strftime("%Y%m%d") + "_e09.nc"
                #                   print(access_path)
                print(access_path)
                if os.path.exists(access_path):
                    #                 for i in range(self.lead_time,self.lead_time+1):
                    #                 for en in ['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']:
                    path = []
                    awap_date = date_time + timedelta(self.lead_time)
                    path.append(date_time)
                    path.append(awap_date)
                    path.append(self.lead_time)
                    _files.append(path)
        return _files

    def __getitem__(self, idx):
        return self.files[idx]
def load_pit_data(lead_time):
    data = aaa(lead_time)
    alpha_lead_time = []
    base_path = "/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/"
    alpha_data = np.load(
        base_path+'alpha_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = dpt.date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        alpha_lead_time.append(alpha_data[idx])
    return np.array(alpha_lead_time, dtype=object)
def calculate_pit_values(ensemble_forecasts, observations, epsilon=1e-6):
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
    ensemble_forecasts = np.array(ensemble_forecasts)
    observations = np.array(observations)
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))
    print("ensemble_forecasts", ensemble_forecasts.shape)
    print("observations", observations.shape)
    # Check input shapes
    if ensemble_forecasts.ndim != 3 or observations.ndim != 2:
        raise ValueError(
            "Input dimensions are incorrect. ensemble_forecasts should be 3D and observations should be 2D.")

    if ensemble_forecasts.shape[:2] != observations.shape:
        raise ValueError("Horizontal and vertical dimensions of ensemble_forecasts and observations must match.")
   
    # Count the total number of ensemble members
    n_ensemble = ensemble_forecasts.shape[-1]
    print("n_ensemble", n_ensemble)
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
    print("pit_values", pit_values.shape)
    return pit_values
def calculate_pit_values_plus(ensemble_forecasts, observations, epsilon=1e-6):
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
    ensemble_forecasts = np.array(ensemble_forecasts)
    observations = np.array(observations)
    ensemble_forecasts = np.transpose(ensemble_forecasts, (1, 2, 0))
    print("ensemble_forecasts", ensemble_forecasts.shape)
    print("observations", observations.shape)
    # Check input shapes
    if ensemble_forecasts.ndim != 3 or observations.ndim != 2:
        raise ValueError(
            "Input dimensions are incorrect. ensemble_forecasts should be 3D and observations should be 2D.")

    if ensemble_forecasts.shape[:2] != observations.shape:
        raise ValueError("Horizontal and vertical dimensions of ensemble_forecasts and observations must match.")
   
    # Count the total number of ensemble members
    n_ensemble = ensemble_forecasts.shape[-1]+1
    print("n_ensemble", n_ensemble)
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
    print("pit_values", pit_values.shape)
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

    return alpha_values

def main(year, time_windows):
    #30year
    Brier_startyear = 1976
    Brier_endyear = 2005
    dates_needs = dpt.date_range(date(year, 1, 1), date(year + 1, 7, 29))
    file_BARRA_dir = "/scratch/iu60/xs5813/Awap_data_bigger/"
    #     date_map=np.array(dates_needs)
    # np.where(date_map==date(2012, 1, 1))
    pit = []

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
                ensamble.append(sr)
                #log_ensamble.append(sr_log)
        if ensamble:
            print("calculate ensemble")
            ensamble = np.array(ensamble)
            print("ensemble.shape:", ensamble.shape)
            print("hr.shape:", hr.shape)
            print("date:", Awap_date)
            pit_value = calculate_pit_values_plus(ensamble, hr)
            pit.append(pit_value)
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
    np.save(os.path.join(base_path, f"alpha_climatology_{year}_all_lead_time_windows_{(time_windows - 1) * 2 + 1}"), np.array(pit, dtype=np.float64))
    #load pit
    for lead_time in range(42):
        pit = load_pit_data(lead_time)
        pit = np.stack(pit, axis=0)
        pit = np.squeeze(pit)
        alpha = calculate_alpha_index(pit)
        print(f"{year}alpha mean",np.mean(alpha))
        file_alpha_bias = 'alpha_climat_lead_time_' + str(lead_time)
        base_path = "/scratch/iu60/xs5813/cli_metric_result/new_crps/save/"
        path_to_create = os.path.join(base_path, norm, "mean_climatology", str(year), "window" + str(window))
        os.makedirs(path_to_create, exist_ok=True)  # exist_ok=True表示如果目录已经存在，不会抛出异常
        np.save(os.path.join(path_to_create, file_alpha_bias), np.array(alpha, dtype=np.float64))

if __name__ == '__main__':
    year_list = [2006]
    timewind = [1]
    for i in timewind:
        for j in year_list:
            main(j, i)
            print(str(j) + ", " + str(i * 2 - 1))
