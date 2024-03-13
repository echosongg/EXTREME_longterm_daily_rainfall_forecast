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
levels["probability"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
enum={0:"0600",1:"1200",2:"1800",3:"0000",4:"0600"}

probability_colours = [
    "#f7fbff",  # Very light blue
    "#deebf7",  # Lighter blue
    "#c6dbef",  # Light blue
    "#9ecae1",  # Medium light blue
    "#6baed6",  # Medium blue
    "#4292c6",  # Medium dark blue
    "#2171b5",  # Dark blue
    "#08519c",  # Darker blue
    "#08306b",  # Very dark blue
    "#041843"   # Darkest blue
]

prcp_colormap = matplotlib.colors.ListedColormap(probability_colours)
#lon_range = (143,  154)
#lat_range = (-32, -24)
lon_range = (142.000,  152.3)
lat_range = (-31.95, -23.4)


##def draw_aus_pr(var,lat,lon,domain = [138, 156.275, -29, -9.975], mode="pr" , titles_on = True,\
#                title = " precipation in 2012", colormap = prcp_colormap, cmap_label = "PR",save=False,path=""):
#def draw_aus_pr(year, month, day, data_type, lat, lon, var, domain=[140.6, 153.9, -39.2, -18.6], title="", \
def draw_aus_pr(year, month, day, leadingtime, lat, lon, var, domain=[142, 152.3, -31.95, -23.8], title="", \
save=False, path="", colormap = prcp_colormap, cmap_label = "PR", mode="probability", titles_on = True):
    """ basema_ploting .py
This function takes a 2D data set of a variable from AWAP and maps the data on miller projection. 
The map default span is longitude between 111.975E and 156.275E, and the span for latitudes is -44.525 to -9.975. 
The colour scale is YlGnBu at 11 levels. 
The levels specifed are suitable for annual rainfall totals for Australia. 
"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap,maskoceans
    
    if mode == "probability":
        level = "probability"
    
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
    
    
    data = xr.DataArray(var,\
                    coords={'lat': lat, 'lon': lon},
                    dims=["lat", "lon"])
    
    cs = map.pcolormesh(x, y, data, norm=BoundaryNorm(level, len(level)-1), cmap=prcp_colormap)
    
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
def main(year, month, day, leadingtime, model_name):
    print("information: crps and expect value")
    base_path = "/scratch/iu60/xs5813/"
    date_string = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    title = None

    file_path = f"{base_path}TestResults/DESRGAN/vTestRefactored/{model_name}/e01/p/{date_string}.nc"
    title = f"probability_e01_{model_name}_data {date_string}"
    # 加载数据
    data = xr.open_dataset(file_path)
    data = data.fillna(0)
    result = data.sel(time=date_string)['p'].values
    lat = data.lat.values
    lon = data.lon.values

    # 绘图
    draw_aus_pr(year, month, day, leadingtime, lat, lon, result, title=title, save=True, path=f"/home/599/xs5813/EXTREME/{leadingtime}_e01_{date_string}_{model_name}.jpeg")

# 例如，要绘制 2002 年 1 月 1 日的预测数据
main(2002, "12", "25", 1, "model_G_i000012_best_20240307-212434")#leadingtime = 1

#p model_G_i000012_best_20240226-005908 效果很好