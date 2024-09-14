#some example of output from model and some result
import xarray as xr
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as func
from mpl_toolkits.basemap import maskoceans
from datetime import date, timedelta, datetime
levels = {}
levels["crps"]   = [0,0.2,0.4,0.6,0.8,1.0] 
levels["crpss"]   = [-0.8,-0.4,-0.2,0,0.2,0.4,0.8] 
#levels["crps"]=[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
levels["new"]   = [0, 0.1, 1.0 ,5.0, 10.0, 20.0, 30.0, 40.0, 60.0 ,100, 150] 
levels["mae"]   = [0, 0.5, 1 ,1.5, 2, 2.5, 3, 4, 6 ,8, 10] 
levels["hour"]  = [0., 0.2, 1, 5,  10,  20,  30,   40,   60,   80,  100,  150]
levels["day"]   = [0., 0.2, 5, 10,  20,  30,  40,  60,  100,  150,  200,  300]
levels["week"]  = [0., 0.2, 10,  20,  30,  50, 100,  150,  200,  300,  500, 1000]
levels["month"] = [0., 10, 20,  30,  40,  50, 100,  200,  300,  500, 1000, 1500]
levels["year"]  = [0., 50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]
enum={0:"0600",1:"1200",2:"1800",3:"0000",4:"0600"}

prcp_colours = [
                   "#FFFFFF", 
                   '#edf8b1',
                   '#c7e9b4',
                   '#7fcdbb',
                   '#41b6c4',
                   '#1d91c0',
                   '#225ea8',
                   '#253494',
                   '#4B0082',
                   "#800080",
                   '#8B0000']

prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)
lon_range = (142.000,  152.3)
lat_range = (-31.95, -23.4)


##def draw_aus_pr(var,lat,lon,domain = [138, 156.275, -29, -9.975], mode="pr" , titles_on = True,\
#                title = " precipation in 2012", colormap = prcp_colormap, cmap_label = "PR",save=False,path=""):
#def draw_aus_pr(year, month, day, data_type, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
#draw_aus_pr(year, month, day, lat, lon,access_lat,access_lon, access_result, qm_result,desrgan_result,prgan_result,awap_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/compare_{ensemble}_{date_string}.jpeg")
def draw_aus_pr(year, month, day, ensemble, lat, lon, access_lat, access_lon, lr, qm, desrgan, prgan, hr, 
                domain=[140.6, 153.9, -39.2, -18.6], title="", save=False, path="", 
                colormap=prcp_colormap, cmap_label="PR", mode="pr", titles_on=True):
    """
    绘制澳大利亚降水（或其他指标）的地图，包含5个并列的图分别显示lr, qm, desrgan, prgan, hr的数据。
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap, maskoceans
    import matplotlib.pyplot as plt
    import numpy as np

    # 设置figure和子图
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 5个并列的图
    # 定义要绘制的变量和数据
    print("drawing")
    data_list = [lr, qm, desrgan, prgan, hr]
    lat_list = [access_lat, lat, lat, lat, lat]  # lr使用access_lat，其余使用lat
    lon_list = [access_lon, lon, lon, lon, lon]  # lr使用access_lon，其余使用lon
    titles = ['LR', 'QM', 'DESRGAN', 'PRGAN', 'HR']

    level = 'new' if mode == 'pr' else ('crpss' if mode == 'crps-ss' else 'crps')
    norm = BoundaryNorm(levels[level], len(levels[level]) - 1)

    for i, ax in enumerate(axes):
        map = Basemap(projection="mill", llcrnrlon=domain[0], llcrnrlat=domain[2], 
                      urcrnrlon=domain[1], urcrnrlat=domain[3], resolution='l', ax=ax)
        map.drawcoastlines()
        map.drawcountries()

        parallels = np.arange(domain[2], domain[3], 5)
        meridians = np.arange(domain[0], domain[1], 5)
        map.drawparallels(parallels, labels=[True, False, False, False] if i == 0 else [False, False, False, False])
        map.drawmeridians(meridians, labels=[False, False, False, True] if i == 4 else [False, False, False, False])

        llons, llats = np.meshgrid(lon_list[i], lat_list[i])
        x, y = map(llons, llats)
        data = xr.DataArray(data_list[i], coords={'lat': lat_list[i], 'lon': lon_list[i]}, dims=["lat", "lon"])

        if mode == "pr" or mode == "mae":
            cs = map.pcolormesh(x, y, data, norm=norm, cmap=colormap)
        elif mode == "crps-ss":
            cs = map.pcolormesh(x, y, data, cmap="RdBu", vmin=-0.8, vmax=0.8)

        if titles_on:
            ax.set_title(titles[i])

    # 添加颜色条，只在最后一个图中显示
    cbar = fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=0.8, extend="max")
    cbar.ax.set_ylabel(cmap_label)

    # 保存或展示图像
    if save:
        plt.savefig(path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
def read_access_data_calibration(root_dir, date_time, leading, year, var_names=["p","alpha","beta"]):
    results = {}
    for var_name in var_names:
        filename = f"{root_dir}/{var_name}/{year}/{date_time}.nc"
        dataset = xr.open_dataset(filename)
        dataset = dataset.fillna(0)
        
        var_data = dataset.isel(time=leading)[var_name].values
        results[var_name] = var_data
        dataset.close()

    return results
def main(year, month, day, ensemble, leadingtime, gtype):
    base_path = "/scratch/iu60/xs5813/"
    date_string = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    date_string2 = f"{year}{month.zfill(2)}{day.zfill(2)}"
    title = None
    desrgan_modelname = "model_G_i000005_20240609-234526"
    prgan_modelname = "model_G_i000008_20240824-212330_with_huber"
    # AWAP 数据的路径
    file_path_awap = f"{base_path}Awap_data_bigger/{date_string}.nc"
    title_awap = f"AWAP_data {date_string}"
    # access_end_time
    access_end_time = datetime.strptime(date_string, "%Y-%m-%d")
    access_start_time_t = access_end_time - timedelta(leadingtime)
    access_start_time = access_start_time_t.strftime("%Y-%m-%d")
    access_start_time2 = access_start_time_t.strftime("%Y%m%d")
    file_path_access = f"{base_path}Processed_data_bigger/{ensemble}/da_pr_{access_start_time2}_{ensemble}.nc"
    title_access = f"ACCESS_data {date_string}"

    #predict data
    #qm
    file_path_qm = f"{base_path}QM_cropped_data/{ensemble}/daq5_pr_{access_start_time2}_{ensemble}.nc"
    #limatology path
    #file_path_clim = 这个part需要问一下，因为和其他的比较起来有所不同，他没有e和leadingtime的概念
    #desrgan, prgan
    def generate_file_path_and_title(base_path, model_name, ensemble, date_string):
        file_path = f"{base_path}TestResults/DESRGAN/vTestRefactored/{model_name}/{ensemble}/pr/{year}/{date_string}.nc"
        title = f"predict_{ensemble}_{model_name}_data {date_string}"
        return file_path, title

    file_path_desrgan, title_desrgan = generate_file_path_and_title(base_path, desrgan_modelname, ensemble, access_start_time)
    file_path_prgan, title_prgan = generate_file_path_and_title(base_path, prgan_modelname, ensemble, access_start_time)
    #read data
    awap_data = xr.open_dataset(file_path_awap)
    awap_data = awap_data.fillna(0)

    access_data = xr.open_dataset(file_path_access)
    access_data = access_data.fillna(0)

    qm_data = xr.open_dataset(file_path_qm, engine='netcdf4')
    qm_data = qm_data.fillna(0)

    desrgan_data = xr.open_dataset(file_path_desrgan)
    desrgan_data = desrgan_data.fillna(0)
    def distribution(file_path_prgan, num_values = 2000):
        prgan_data = xr.open_dataset(file_path_prgan)
        
        if gtype == "mean":
            prgan_data = prgan_data.isel(time=leadingtime)['pr']
            print(prgan_data)

        elif gtype == "median":
            file_path_prgan = re.sub(r'pr/.*$', '', file_path_prgan)
            # 替换 'pr' 生成新的 file_path
            print(file_path_prgan)
            sr = read_access_data_calibration(
                file_path_prgan, access_start_time, leadingtime, year, ["p","alpha", "beta"])
            # 初始化预测数组
            def median(p_pred, alpha_pred, beta_pred, num_values):
                alpha_pred = np.clip(np.array(alpha_pred, dtype=np.float32), a_min=1e-6, a_max=None)
                beta_pred = np.clip(np.array(beta_pred, dtype=np.float32), a_min=1e-6, a_max=None)#beta_pred都是0
                # Printing shape, values, and attributes of beta_pred
                p_pred = np.array(p_pred, dtype=np.float32)
                forecasts = np.zeros((num_values, *p_pred.shape), dtype=np.float32)
                
                for i in range(num_values):
                    #is_rain = (p_pred > 0.5).astype(int)
                    is_rain = np.random.binomial(1, p_pred)  # 使用二项分布模拟 Bernoulli 过程
                    rain_amount = np.random.gamma(alpha_pred, 1/beta_pred)  # 生成 gamma 分布的随机数
                    forecasts[i] = is_rain * rain_amount
                forecasts = forecasts.reshape(-1, *p_pred.shape)  # 调整形状
                median_forecasts = np.median(forecasts, axis=0)
                return median_forecasts
            prgan_data = median(np.squeeze(sr['p']), np.squeeze(sr['alpha']), np.squeeze(sr['beta']), num_values)#none
        return prgan_data
    #prgan_data = xr.open_dataset(file_path_prgan)
    prgan_data = distribution(file_path_prgan)
    print("prgan",prgan_data)
    
    #get the lat and lon for different resolution
    access_lat = access_data.lat.values
    access_lon = access_data.lon.values

    lat = awap_data.lat.values
    lon = awap_data.lon.values
    #get result
    awap_result = awap_data.isel(time=0)['precip'].values
    access_result = access_data.isel(time=leadingtime)['pr'].values * 86400
    print("access_result", access_result)
    desrgan_result = desrgan_data.isel(time=leadingtime)['pr'].values
    prgan_result = prgan_data
    qm_result = qm_data.isel(time=leadingtime)['pr'].values
    #def draw_aus_pr(year, month, day, esemeble, lat, lon, var, domain=[142, 152.3, -31.95, -23.8], title="", \
#save=False, path="", colormap = prcp_colormap, cmap_label = "PR", mode="pr", titles_on = True):
    draw_aus_pr(year, month, day, ensemble, lat, lon,access_lat,access_lon, access_result, qm_result,desrgan_result,prgan_result,awap_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/compare_{gtype}_{ensemble}_{date_string}_{leadingtime}.jpeg")

main(2009, "08", "31", "e01", 1, "median")
main(2009, "08", "31", "e01", 1, "mean")