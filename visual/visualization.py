#some example of output from model and some result
import xarray as xr
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
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
#lon_range = (143,  154)
#lat_range = (-32, -24)
lon_range = (142.000,  152.3)
lat_range = (-31.95, -23.4)


##def draw_aus_pr(var,lat,lon,domain = [138, 156.275, -29, -9.975], mode="pr" , titles_on = True,\
#                title = " precipation in 2012", colormap = prcp_colormap, cmap_label = "PR",save=False,path=""):
#def draw_aus_pr(year, month, day, data_type, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
def draw_aus_pr(year, month, day, esemeble, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
save=False, path="", colormap = prcp_colormap, cmap_label = "PR", mode="pr", titles_on = True):
    """ basema_ploting .py
This function takes a 2D data set of a variable from AWAP and maps the data on miller projection. 
The map default span is longitude between 111.975E and 156.275E, and the span for latitudes is -44.525 to -9.975. 
The colour scale is YlGnBu at 11 levels. 
The levels specifed are suitable for annual rainfall totals for Australia. 
"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap,maskoceans
    
    if mode == "pr":
        level = 'new'
    
    # crps-ss
    if mode == "crps-ss":
        level = "crpss"
             
    if mode == "crps":
        level = "crps"  
    
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
    
    # pr
    if mode == "pr":
        cs = map.pcolormesh(x, y, data, norm = norm, cmap = colormap) 
    
    # crps-ss
    if mode == "crps-ss":
        cs = map.pcolormesh(x, y, data, cmap="RdBu",vmin=-0.8,vmax=0.8) 
        
    if mode == "mae":
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
        plt.savefig(path, bbox_inches = 'tight', dpi=300)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
    return

# 主要的函数，用来根据年月日和数据类型加载数据并绘图
def main(year, month, day, esemeble, model_name, leadingtime):
    print("information: crps and expect value")
    base_path = "/scratch/iu60/xs5813/"
    date_string = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    title = None
    #predict data
    file_path_predict = f"{base_path}TestResults/DESRGAN/vTestRefactored/{model_name}/{esemeble}/pr/{year}/{date_string}.nc"
    title_predict = f"predict_{esemeble}_{model_name}_data {date_string}"
    # AWAP 数据的路径
    file_path_awap = f"{base_path}Awap_data_bigger/{date_string}.nc"
    title_awap = f"AWAP_data {date_string}"

    access_end_time = datetime.strptime(date_string, "%Y-%m-%d")
    access_start_time_t = access_end_time - timedelta(leadingtime)
    access_start_time = access_start_time_t.strftime("%Y-%m-%d")
    access_start_time2 = access_start_time_t.strftime("%Y%m%d")
    file_path_access = f"{base_path}Processed_data_train/{esemeble}/da_pr_{access_start_time2}_{esemeble}.nc"
    title_access = f"ACCESS_data {date_string}"
    #predict_data = xr.open_dataset(file_path_predict)
    #predict_data = predict_data.fillna(0)

    awap_data = xr.open_dataset(file_path_awap)
    awap_data = awap_data.fillna(0)

    access_data = xr.open_dataset(file_path_access)
    access_data = access_data.fillna(0)
        # 预测数据的路径
    #predict_result = predict_data.sel(time=date_string)['pr'].values

    #awap_result = awap_data['pr'].values
    access_result = np.expm1(access_data.isel(time=leadingtime)['pr'].values * 7)
    #result = np.expm1(data.sel(time=date_string)['pr'].values * 7)
    #lat = predict_data.lat.values
    #lon = predict_data.lon.values

    access_lat = access_data.lat.values
    access_lon = access_data.lon.values

    #draw_aus_pr(year, month, day, "predict", lat, lon, predict_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/predict_e01_{date_string}_{model_name}.jpeg")
    #draw_aus_pr(year, month, day, "awap", lat, lon, awap_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/awap_e01_{date_string}_{model_name}.jpeg")
    draw_aus_pr(year, month, day, "access", access_lat, access_lon, access_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/access_e01_{date_string}_{leadingtime}.jpeg")

# 例如，要绘制 2002 年 1 月 1 日的预测数据
main(2005, "12", "31", "e01", "model_G_i000008_20240824-212330_with_huber", 1)
