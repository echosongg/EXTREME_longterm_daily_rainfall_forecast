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
def draw_aus_pr(year, month, day, esemeble, lat, lon, var, domain=[142, 152.3, -31.95, -23.8], title="", \
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
    file_path_predict = f"{base_path}TestResults/DESRGAN/vTestRefactored/{model_name}/{esemeble}/pr/{date_string}.nc"
    title_predict = f"predict_{esemeble}_{model_name}_data {date_string}"
    # AWAP 数据的路径
    file_path_awap = f"{base_path}Awap_pre_data/{date_string}.nc"
    title_awap = f"AWAP_data {date_string}"
     # ACCESS 数据的

    access_end_time = datetime.strptime(date_string, "%Y-%m-%d")
    access_start_time = access_end_time - timedelta(leadingtime)
    access_start_time = access_start_time.strftime("%Y-%m-%d")
    file_path_access = f"{base_path}Processed_data/{esemeble}/{access_start_time}.nc"
    title_access = f"ACCESS_data {date_string}"

    # 加载数据
    predict_data = xr.open_dataset(file_path_predict)
    predict_data = predict_data.fillna(0)

    awap_data = xr.open_dataset(file_path_awap)
    awap_data = awap_data.fillna(0)

    access_data = xr.open_dataset(file_path_access)
    access_data = access_data.fillna(0)
        # 预测数据的路径
    predict_result = np.expm1(predict_data.sel(time=date_string)['pr'].values * 7)

    awap_result = awap_data['pr'].values
    access_result = access_data.isel(time=leadingtime)['pr'].values
    #result = np.expm1(data.sel(time=date_string)['pr'].values * 7)
    lat = predict_data.lat.values
    lon = predict_data.lon.values

    access_lat = access_data.lat.values
    access_lon = access_data.lon.values

    draw_aus_pr(year, month, day, "predict", lat, lon, predict_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/predict_e01_{date_string}_{model_name}.jpeg")
    draw_aus_pr(year, month, day, "awap", lat, lon, awap_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/awap_e01_{date_string}_{model_name}.jpeg")
    draw_aus_pr(year, month, day, "access", access_lat, access_lon, access_result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/access_e01_{date_string}_{leadingtime}.jpeg")

# 例如，要绘制 2002 年 1 月 1 日的预测数据
main(2002, "12", "27", "e01", "model_G_i000005_best_20240305-194851", 1)
#1994-07-16

#not good data
#gernerator loss: log loss -> model_G_i000003_best_20240125-022940
#model_G_i000004_best_20240203-091026 预测雨量集中在海边
#model_G_i000003_best_20240204-195122 貌似位置是对的，就是值差的很多
#model_G_i000007_best_20240205-034204 crps and expect value, 左边一条浅色竖线
#model_G_i000003_best_20240213-090639 中间测试一下 crps and huber 1e-2, 位置不好在训练看看
#model_G_i000010_best_20240213-093949 change D using log loss
#model_G_i000008_best_20240214-161433 pretrain L1 1 L2 1 还没试
#model_G_i000006_best_20240214-123451  pretrain L1 1 L2 1很烂

#** model_G_i000010_best_20240214-125812 pretrain L1 1, L2 0.1 这个还好，有一些位置有一点奇怪
#model_G_i000013_best_20240214-163449 original DESRGAN 看看效果如何 不是很好有点过拟合
#model_G_i000006_20240202-130814 这个是original 看着效果不好，但是用作pretrain还可以
#20240214-160606 这个还可以 original model
#model_G_i000010_best_20240214-105741 pretrain L1 0.1, L2 1
#model_G_i000007_best_20240217-145726 正在跑omodel的 3 esemeble
#model_G_i000005_best_20240217-153636 3 esemeble not bad，继续训练看看结果可不可以更好, 正在跑
#model_G_i000010_best_20240218-015617 9 esemeble 
#这两个都可以试一试
#model_G_i000004_best_20240219-114819 summer 3 esemble 还可以
#model_G_i000010_best_20240219-121948 summer 3 e 效果貌似不错
#model_G_i000001_best_20240219-152044 summer 3 e 好烂
#model_G_i000005_best_20240221-003227 no tensor log loss一片空白 
#model_G_i000003_best_20240220-191719 27-2 tensor 
#model_G_i000004_best_20240221-231804 0.1 log loss (1-p) a/b的generate, cut了0值，结果没什么白色，但是颜色貌似是对的


#model_G_i000004_best_20240223-002852 0.001 log loss no huber 效果不错，是改了之后的
#model_G_i000007_20240223-010200 0.001 log loss and huber 效果还行
#model_G_i000010_best_20240223-025648 这是目前效果最好的，但是边上会有一些问题，还有就是极值问题，老师让我将其改为输出三个值，p, a, b然后看看预测的怎么样
#这个e01, e02, e03 summer, leadingtime = 1-7
#

#1. 我输出的结果似乎在极值方面做的不是很好，和original_model相比，有雨的部分会shrink -> 由于NewZeland文章是使用均值，
#   所以我将其改为均值，似乎是这个原因？解决想法： 1mm too big, make it smaller, 改善预处理方法也可能可以对极值有改善（由于极值对huber和log loss的影响比较小， 原因是被log拉低了）
#2. 似乎因为很多我的地区都是没有雨的，导致模型经常会趋向于全图无雨, 改为预测夏季
#3. 不论是我的模型还是DESGAN 在边远的部分会有一些竖线，估计是因为在边上，随机选取块的时候会比较训练的少，大部分时候取中间
#4. 

