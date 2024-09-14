#some example of output from model and some result
#average from esembles
import xarray as xr
import matplotlib
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.basemap import maskoceans
from datetime import date, timedelta, datetime
import torch
import matplotlib.colors as mcolors
print(torch.__version__)

levels = {}
#levels["crps"]   = [0,0.2,0.4,0.6,0.8,1.0] 
levels["crpss"]   = [-0.8,-0.4,-0.2,0,0.2,0.4,0.8] 
#levels["crps"]=[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
levels["new"]   = [0, 0.1, 1.0 ,5.0, 10.0, 20.0, 30.0, 40.0, 60.0 ,100, 150] 
levels["mae"]   = [0, 0.5, 1 ,1.5, 2, 2.5, 3, 4, 6 ,8, 10] 
levels["hour"]  = [0., 0.2, 1, 5,  10,  20,  30,   40,   60,   80,  100,  150]
levels["day"]   = [0., 0.2, 5, 10,  20,  30,  40,  60,  100,  150,  200,  300]
levels["week"]  = [0., 0.2, 10,  20,  30,  50, 100,  150,  200,  300,  500, 1000]
levels["month"] = [0., 10, 20,  30,  40,  50, 100,  200,  300,  500, 1000, 1500]
levels["year"]  = [0., 50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]
levels["crps"]  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
enum={0:"0600",1:"1200",2:"1800",3:"0000",4:"0600"}

prcp_colours = [
    "#FFFFFF",  # 保持白色，表示无误差或最小误差
    "#fff5f0",  # 非常浅的红色
    "#ffe4e1",  # 更浅的红色
    "#ffd5d4",  # 更浅的红色
    "#ffc6c6",  # 更浅的红色
    "#fcbba1",  # 中等浅红色
    "#fcaea1",  # 中等浅红色
    "#fc9272",  # 亮红色
    "#fb6a4a",  # 中红色
    "#f74234",  # 中等红色
    "#ef3b2c",  # 深红色
    "#cb181d",  # 更深的红色
    "#a50f15",  # 暗红色
    "#800012",  # 更深的暗红色
    "#67000d",  # 非常深的红色
    "#49000a"   # 极深的红色
]


prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)
#lon_range = (143,  154)
#lat_range = (-32, -24)
lon_range = (140.6, 153.9)
lat_range = (-39.2, -18.6)
#australia_lon_range = (142.000,  152.3)
#australia_lat_range = (-31.95, -23.4)

##def draw_aus_pr(var,lat,lon,domain = [138, 156.275, -29, -9.975], mode="pr" , titles_on = True,\
#                title = " precipation in 2012", colormap = prcp_colormap, cmap_label = "PR",save=False,path=""):
#def draw_aus_pr(year, month, day, data_type, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
def draw_aus(year, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
save=False, path="", colormap = prcp_colormap, cmap_label = "MAE", mode="mae", titles_on = True):
    """ basema_ploting .py
This function takes a 2D data set of a variable from AWAP and maps the data on miller projection. 
The map default span is longitude between 111.975E and 156.275E, and the span for latitudes is -44.525 to -9.975. 
The colour scale is YlGnBu at 11 levels. 
The levels specifed are suitable for annual rainfall totals for Australia. 
"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap,maskoceans
    level = "aa"
    print("var_shape",var.shape)
    if mode == "pr":
        level = 'new'
    
    # crps-ss
    if mode == "crps-ss":
        level = "crpss"
             
    if mode == "skil_dis" or mode == "skil":
        level = "crps"  

    if mode == "mae" or mode == "mae_mean" or mode == "mae_median" or mode == "rmse":
        level = "mae"
    
    fig=plt.figure()
    level=levels[level]
    map = Basemap(projection="mill", llcrnrlon=domain[0], llcrnrlat=domain[2], urcrnrlon=domain[1],
                  urcrnrlat=domain[3], resolution='l')
    map.drawcoastlines()
    map.drawcountries()

    parallels = np.arange(domain[2], domain[3], 5)  # 纬度间隔 5 度
    map.drawparallels(parallels, labels=[True, False, False, False])  # 只在左侧添加纬度标签
    meridians = np.arange(domain[0], domain[1], 5)  # 经度间隔 5 度
    map.drawmeridians(meridians, labels=[False, False, False, True])  # 只在底部添加经度标签


    llons, llats = np.meshgrid(lon, lat)
    x, y = map(llons, llats)
    print(x.shape,y.shape)
    
    norm = BoundaryNorm(level, len(level)-1)
    
    # red square
    #var[255:260,205:510]= 1000
    #var[495:500,210:510]= 1000
    #var[260:500,205:210]= 1000
    #var[260:500,505:510]= 1000
    
    data = xr.DataArray(var,\
                    coords={'lat': lat, 'lon': lon},
                    dims=["lat", "lon"])
    #print("结果的形状:", var.shape)

    # 选择第一个时间步骤的数据
    #selected_result = var[0, :, :]
    #print("time", var[:,0,0])

    # 现在使用选择的数据创建 DataArray
    #data = xr.DataArray(selected_result, coords=[lat, lon], dims=["lat", "lon"])
    #print("data",data)
    cmap = mcolors.LinearSegmentedColormap.from_list('white_to_red', ['#FFFFFF', '#FF0000'])
    # pr
    if mode == "pr":
        cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
    
    # crps-ss
    if mode == "crps-ss":
        cs = map.pcolormesh(x, y, data, cmap="RdBu",vmin=-0.8,vmax=0.8) 
        
    if mode == "mae" or mode == "mae_mean" or mode == "mae_median" or mode == "rmse":
        cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
    if mode == "skil_dis" or "skil":
        cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
        
    
    if titles_on:
        # label with title, latitude, longitude, and colormap
        
        plt.title(title)
#         x = [115, 120, 125, 130, 135, 140, 145, 150, 155]
#         x_label = ['115E', '120E', '125E', '130E', '135E', '140E', '145E', '150E','155E']
#         plt.xticks(x, x_label, rotation ='vertical')
        plt.xlabel("\n\n\nLongitude")
        plt.ylabel("Latitude\n\n")
        
        # color bar
        cbar = plt.colorbar(ticks = level[:-1], shrink = 0.8, extend = "max")#shrink = 0.8
        cbar.ax.set_ylabel(cmap_label)
        
        #cbar.ax.set_xticklabels(level) #报错
    
    # plt.plot([-1000,1000],[900,1000], c="b", linewidth=2, linestyle=':')
    
    if save:
        if 'path' not in locals():  # 检查path是否已定义
            path = 'default_path.png'  # 设定默认路径
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)  # 创建路径（如果需要）
        plt.savefig(path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
    return
# def print_by_lead(base_path, model_name, year, bool_print):
#     if(not bool_print):
#         return
#     metric_name = {'bias',  'bias_median', 'Brier_0',  'Brier_95',  'Brier_99',  'mae',  'mae_mean',  'mae_median',  'relative_bias',  'rmse',  'skil'}
    
#     for leading in range(0, 41):
#         for metric in metric_name:
#             file_path_metric = f"{base_path}{metric}/{model_name}/{year}/lead_time{leading}_whole.npy"
#             metric_data = np.load(file_path_metric)
#             metric_data = metric_data
#             metric_mean = np.mean(metric_data)
#             print(f"Mean of metric data({year}_{metric}_{model_name}_leadingtime{leading}):", metric_mean)

# 主要的函数，用来根据年月日和数据类型加载数据并绘图
def main(year, model_name, leadingtime, metric, QM = True, climatology = True):
    print("information: crps and expect value")
    base_path = "/scratch/iu60/xs5813/metric_results/"
    title = None
    #predict data
    #file_path = '/scratch/iu60/xs5813/metric_results/mae_mean/model_G_i000005_20240331-075814/2007/lead_time0_whole.npy'
    file_path_metric = f"{base_path}{metric}/{model_name}/{year}/lead_time{leadingtime}_whole.npy"
    title = f"{metric} data visual"
    file_path_awap = "/scratch/iu60/xs5813/Awap_data_bigger/2001-01-15.nc"
    #just to het lat and lon
    awap_data = xr.open_dataset(file_path_awap)
    # 加载数据
    metric_data = np.load(file_path_metric)
    metric_data = metric_data
    metric_mean = np.mean(metric_data)
    print(f"Mean of metric data({year}_{metric}_{model_name}_leadingtime{leadingtime}):", metric_mean)
    lat = awap_data.lat.values
    lon = awap_data.lon.values
    #print_by_lead(base_path, model_name, year, True)
    draw_aus(year, lat, lon, metric_data, title=title, save=True, path=f"/home/599/xs5813/PEFGAN-for-rainfall/visualization/data_value/{year}_{metric}_{model_name}_leadingtime{leadingtime}.jpeg", colormap = prcp_colormap, cmap_label = metric, mode=metric)
#def draw_aus(year, month, day, esemeble, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
#save=False, path="", colormap = prcp_colormap, cmap_label = "MAE", mode="mae", titles_on = True):
    qm_base = f"/scratch/iu60/xs5813/qm/new_crps/save/crps_ss/QM/{year}/lead_time{leadingtime}_whole.npy"
    clim_base = f"/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/mean_climatology/{year}/window1/climat_lead_time_{leadingtime}.npy"
# 例如，要绘制 2002 年 1 月 1 日的预测数据
    qm_title = f"crps data visual from QM in {year} at leaingtime{leadingtime}"
    qm_metric_data = np.load(qm_base)
    qm_metric_mean = np.mean(qm_metric_data)
    draw_aus(year, lat, lon, qm_metric_data, title=qm_title, save=True, path=f"/home/599/xs5813/PEFGAN-for-rainfall/visualization/data_value/{year}_{metric}_QM_leadingtime{leadingtime}.jpeg", colormap = prcp_colormap, cmap_label = metric, mode=metric)
    clim_title = f"crps data visual from climatology in {year} at leaingtime{leadingtime}"
    clim_metric_data = np.load(clim_base)
    clim_metric_mean = np.squeeze(np.mean(clim_metric_data))
    draw_aus(year, lat, lon, clim_metric_data, title=clim_title, save=True, path=f"/home/599/xs5813/PEFGAN-for-rainfall/visualization/data_value/{year}_{metric}_QM_leadingtime{leadingtime}.jpeg", colormap = prcp_colormap, cmap_label = metric, mode=metric)
#main(2007, "12", "27", "e01", "model_G_i000006_20240402-002610", 1)
main(2006,"model_G_i000005_20240403-035316_with_huber", 1, "skil_dis")#leadingtime = 1

