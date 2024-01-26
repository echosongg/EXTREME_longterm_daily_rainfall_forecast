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
lon_range = (142.975,  154.275)
lat_range = (-31.525, -23.975)


##def draw_aus_pr(var,lat,lon,domain = [138, 156.275, -29, -9.975], mode="pr" , titles_on = True,\
#                title = " precipation in 2012", colormap = prcp_colormap, cmap_label = "PR",save=False,path=""):
def draw_aus_pr(year, month, day, data_type, lat, lon, var, domain=[142.975, 154.275, -31.525, -23.975], title="", \
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

    parallels = np.arange(domain[2], domain[3], 2)  # 纬度间隔 5 度
    map.drawparallels(parallels, labels=[True, False, False, False])  # 只在左侧添加纬度标签
    meridians = np.arange(domain[0], domain[1], 2)  # 经度间隔 5 度
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
def main(year, month, day, data_type):
    base_path = "/scratch/iu60/xs5813/"
    date_string = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    title = None
    
    if data_type == "predict":
        # 预测数据的路径
        file_path = f"{base_path}TestResults/DESRGAN/vTestRefactored/model_G_i000003_best_20240125-022940/e01/{date_string}.nc"
        title = f"predict_e01_data {date_string}"
    elif data_type == "awap":
        # AWAP 数据的路径
        file_path = f"{base_path}Awap_pre_data/{date_string}.nc"
        title = f"AWAP_data {date_string}"
    elif data_type == "access":
        # ACCESS 数据的路径
        file_path = f"{base_path}Processed_data/e02/{date_string}.nc"
        title = f"ACCESS_data {date_string}"

    # 加载数据
    data = xr.open_dataset(file_path)
    data = data.fillna(0)
    if data_type == "predict":
        # 预测数据的路径
        result = np.expm1(data.sel(time=date_string)['pr'].values * 7)
    elif data_type == "awap":
        result = data['pr'].values
    elif data_type == "access":
        # ACCESS 数据的路径
        result = data.sel(time=date_string)['pr'].values
    #result = np.expm1(data.sel(time=date_string)['pr'].values * 7)
    lat = data.lat.values
    lon = data.lon.values

    # 绘图
    draw_aus_pr(year, month, day, data_type, lat, lon, result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/{data_type}_e01_{date_string}.jpeg")

# 例如，要绘制 2002 年 1 月 1 日的预测数据
main(2002, "01", "25", "predict")
#1994-07-16

#not good data
#gernerator loss: log loss -> model_G_i000003_best_20240125-022940

