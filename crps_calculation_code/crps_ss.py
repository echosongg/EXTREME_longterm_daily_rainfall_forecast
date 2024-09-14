import xarray as xr
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap, maskoceans
import csv
model_name = "model_G_i000006_20240610-011512"
#model_name = "model_G_i000008_20240824-212330_with_huber"
# 评估函数
def evaluate(year, lead, draw=False, window=1):

    print(f"{year} , {lead}")
    print('Outputing the csv file....')
    print(f'Model name is: {model_name}')

    
    total_crps_list = []
    
    qm_crps_list = []
    #qm_mask = np.load("/scratch/iu60/yl3101/QM(AWAP)_mask/awap_binary_mask.npy")
    # 定义经纬度范围
    lon_min, lon_max = 140.6, 153.9
    lat_min, lat_max = -39.2, -18.6

    # 生成经纬度网格
    lon = np.linspace(lon_min, lon_max, 267)  # 根据需要调整分辨率
    lat = np.linspace(lat_min, lat_max, 413)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 创建Basemap对象
    m = Basemap(projection='mill', llcrnrlon=lon_min, llcrnrlat=lat_min,
                urcrnrlon=lon_max, urcrnrlat=lat_max, resolution='l')

    # 创建海洋掩码
    mask = maskoceans(lon_grid, lat_grid, np.ones(lon_grid.shape), inlands=True)

    # 提取掩码数组
    qm_mask = mask.mask

    paths = [
        f"./csv_files/{model_name}/{year}/",
        f"./image/QM/{year}/window_{window}",
        f"./image/{model_name}/{year}/window_{window}"
    ]

    for path in paths:
        os.makedirs(path, exist_ok=True)

    model_file_csv = f"{paths[0]}/crps_ss_{year}_window{window}.csv"
    climat_file_csv = f"{paths[0]}/climatology_{year}_window{window}.csv"
    QM_file_csv = f"{paths[0]}/QM_{year}_window{window}.csv"
    print("aa")

    def write_csv(file_path, header, data):
        with open(file_path, "w", newline='') as file:
            csv_file = csv.writer(file)
            csv_file.writerow(header)
            for line in data:
                csv_file.writerow(line)

    def process_climatology_data(time):
        base_path = "/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/mean_climatology"
        metrics = [
            "climat_lead_time_","mae_median_climat_lead_time_","mae_climat_lead_time_","bias_climat_lead_time_", "bias_median_climat_lead_time_", "relative_bias_climat_lead_time_",
            "prob95_climat_lead_time_", "prob99_climat_lead_time_", "prob995_climat_lead_time_"
        ]
        data = []
        for metric in metrics:
            path = f"{base_path}/{year}/window{window}/{metric}{time}.npy"
            metric_data = np.load(path, allow_pickle=True).astype("float32")
            metric_data = np.ma.masked_array(metric_data, mask=qm_mask)
            data.append(np.nanmean(metric_data))
        return data

    climat_data = [[time, *process_climatology_data(time)] for time in range(lead)]
    write_csv(climat_file_csv, ["lead time", "crps", "mae_median","mae_mean","bias", "bias_median","relative_bias", "Brier95", "Brier99","Brier995"], climat_data)

    def process_qm_data(time):
        #qm_base = "/scratch/iu60/hs2870/score/save/"
        qm_base = "/scratch/iu60/xs5813/qm/new_crps/save/"
        total_qm = np.zeros((413, 267))  # Initialize total_qm here
        paths = [
            f"{qm_base}crps_ss/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}mae_median/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}mae/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}bias/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}bias_median/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}relative_bias_5/QM/{year}/lead_time{time}_whole.npy",
           #f"{qm_base}Brier0/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}Brier95/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}Brier99/QM/{year}/lead_time{time}_whole.npy",
            f"{qm_base}Brier995/QM/{year}/lead_time{time}_whole.npy",
            #f"{qm_base}heavy30/QM/{year}/lead_time{time}_whole.npy",
            #f"{qm_base}Brier999/QM/{year}/lead_time{time}_whole.npy",
        ]
        data = []
        for path in paths:
            metric_data = np.load(path, allow_pickle=True).astype("float32")
            print("qm path", path)
            metric_data = np.ma.masked_array(metric_data, mask=qm_mask)
            data.append(np.nanmean(metric_data))
        leading0_qm = np.load(paths[0])
        base_path = "/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/"
        leading0_clima = np.load(f"{base_path}mean_climatology/{year}/window{window}/climat_lead_time_{time}.npy", allow_pickle=True)
        leading0_clima = leading0_clima.astype("float32")
        leading0_clima = np.ma.masked_array(leading0_clima, mask=qm_mask)
        crpsss_qm = 1 - (leading0_qm / leading0_clima)
        qm_crps_list.append(crpsss_qm)
        total_qm = total_qm + crpsss_qm
        data.insert(0, np.nanmean(crpsss_qm))
        #data.insert(0, np.nanmean(leading0_qm))
        return data

    qm_data = [[time, *process_qm_data(time)] for time in range(lead)]
    write_csv(QM_file_csv, ["lead time", "crps_ss", "crps","mae_median","mae_mean","bias","bias_median", "relative_bias_5","Brier95", "Brier99","Brier995"], qm_data)

    def process_model_data(time):
        model_base_path = f"/scratch/iu60/xs5813/metric_results/"
        #model_name = "model_G_i000006_20240401-042157"
        paths = [
            # f"{model_base_path}skil/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}mae_median/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}mae_mean/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}bias/{model_name}/{year}/lead_time{time}_whole.npy",
            # #f"{model_base_path}bias_median/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}relative_bias_5/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}Brier_95/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}Brier_99/{model_name}/{year}/lead_time{time}_whole.npy",
            # f"{model_base_path}Brier_995/{model_name}/{year}/lead_time{time}_whole.npy",
            f"{model_base_path}alpha/{model_name}/{year}/lead_time{time}_whole.npy",
            #f"{model_base_path}Brier_0_dis/{model_name}/{year}/lead_time{time}_whole.npy",
            #f"{model_base_path}Brier_95_dis/{model_name}/{year}/lead_time{time}_whole.npy",
            #f"{model_base_path}Brier_99_dis/{model_name}/{year}/lead_time{time}_whole.npy",
            #f"{model_base_path}Brier_995_dis/{model_name}/{year}/lead_time{time}_whole.npy",
            #distribution
            #f"{model_base_path}skil_dis/{model_name}/{year}/lead_time{time}_whole.npy",
#"lead time", "crpsss", "crps", "mae_median","mae_mean","bias","bias_median","relative_bias","relative_bias_2","relative_bias_4","relative_bias_5", "Brier0", "Brier95", "Brier99","Brier995", "Brier95dis", "Brier99dis","Brier995dis"
        ]
        data = []
        for path in paths:
            metric_data = np.load(path, allow_pickle=True).astype("float32")
            print("path", path)
            print("metric_data", metric_data.shape)
                        # Check if path contains '_dis' and apply double nanmean if it does
            # if 'mae_median' in path:
            #     metric_data = np.ma.masked_array(metric_data, np.tile(qm_mask, (9, 1, 1)))
            #     mean_value = np.nanmean(metric_data)
            # else:
            #     metric_data = np.ma.masked_array(metric_data, mask=qm_mask)
            #     mean_value = np.nanmean(metric_data)
            metric_data = np.ma.masked_array(metric_data, mask=qm_mask)
            data.append(np.nanmean(metric_data))
        leading0_v3 = np.load(paths[0])
        base_path = "/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/"
        leading0_clima = np.load(f"{base_path}mean_climatology/{year}/window{window}/climat_lead_time_{time}.npy", allow_pickle=True)
        leading0_clima = leading0_clima.astype("float32")
        leading0_clima = np.ma.masked_array(leading0_clima, mask=qm_mask)
        crpsss_v3 = 1 - (leading0_v3 / leading0_clima)
        total_crps_list.append(crpsss_v3)
        total = np.zeros((413, 267))
        total = total + crpsss_v3
        data.insert(0, np.nanmean(crpsss_v3))
        #data.insert(0, np.nanmean(leading0_v3))
        return data

    model_data = [[time, *process_model_data(time)] for time in range(lead)]
    write_csv(model_file_csv, ["lead time", "crpsss", "crps", "mae_median","mae_mean","bias","relative_bias_5", "Brier95", "Brier99","Brier995","alpha"], model_data)

# 运行评估函数
years = ['2006']
for year in years:
    evaluate(year, 42, window=1, draw=False)
print(model_name,years, "done")
