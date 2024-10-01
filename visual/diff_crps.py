#写一个函数，参数：年份， leadingtime，选择对比的方法（QM， DESRGAN， PEFGAN）, brier 9*
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.basemap import maskoceans

colors = [
    (0, 0, 0.5),      # 深蓝色
    (0, 0, 0.75),     # 蓝色
    (0, 0, 1),        # 浅蓝色
    (0.3, 0.3, 1),    # 更浅的蓝色
    (0.5, 0.5, 1),    # 更更浅的蓝色
    (0.75, 0.75, 1),  # 非常浅的蓝色
    (1, 0.75, 0.75),  # 非常浅的红色
    (1, 0.5, 0.5),    # 更更浅的红色
    (1, 0.3, 0.3),    # 更浅的红色
    (1, 0, 0),        # 红色
    (0.75, 0, 0),     # 深红色
    (0.5, 0, 0)       # 深深红色
]
levels = {}
#   5th Percentile: 3.0064114753258764e-08
#   50th Percentile: 0.09492528438568115
#   95th Percentile: 0.6774376034736633
levels["skil"] = [-1, -0.5, -0.3, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.3, 0.5, 1]
#0.0094
def draw(my_model_data, compare_data, lon, lat, title):
    level = levels["skil"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=len(level))
    data_diff = xr.DataArray(
        compare_data - my_model_data,
        coords={'lat': lat, 'lon': lon},
        dims=["lat", "lon"]
    ).astype("float32")
    # 统计正值的点数
    positive_diff_count = np.sum(data_diff >= 0)
    total_count = data_diff.size
    positive_percentage = (positive_diff_count / total_count) * 100
    print("percentage better than that",float(positive_percentage))
    print(f"  Mean: {np.mean(np.abs(data_diff))}")
    print(f"  Std: {np.std(np.abs(data_diff))}")
    print(f"  Min: {np.min(np.abs(data_diff))}")
    print(f"  Max: {np.max(np.abs(data_diff))}")
    print(f"  5th Percentile: {np.percentile(np.abs(data_diff), 5)}")
    print(f"  50th Percentile: {np.percentile(np.abs(data_diff), 50)}")
    print(f"  95th Percentile: {np.percentile(np.abs(data_diff), 95)}")

    # 设置绘图和颜色映射
    fig = plt.figure(figsize=(10, 6))
    m = Basemap(projection='cyl', llcrnrlon=lon.min(), llcrnrlat=lat.min(),
                urcrnrlon=lon.max(), urcrnrlat=lat.max(), resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    
    # 绘制纬度和经度标记
    parallels = np.arange(int(lat.min()), int(lat.max()), 5.0)
    m.drawparallels(parallels, labels=[True, False, False, False])
    meridians = np.arange(int(lon.min()), int(lon.max()), 5.0)
    m.drawmeridians(meridians, labels=[False, False, False, True])

    # 转换坐标系
    x, y = m(*np.meshgrid(lon, lat))
    
    # 设置颜色映射和规范化
    norm = BoundaryNorm(level, len(level)-1)  #
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=len(level))
    
    # 数据掩蔽海洋部分
    masked_data = maskoceans(x, y, data_diff, resolution='c', grid=1.25)

    # 绘制差异
    cs = m.pcolormesh(x, y, masked_data, cmap=cmap, norm=norm)
    
    # 添加色标
    cbar = m.colorbar(cs, location='right', pad="10%")
    cbar.set_label('Difference')
    
    # 设置标题和标签
    plt.title("")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 保存图片
    save_path = f"/home/599/xs5813/EXTREME/visual/{title.replace('/', '_')}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def diff(year, leadingtime, method="QM"):
    # 获取纬度和经度信息

    file_path_awap = "/scratch/iu60/xs5813/Awap_data_bigger/2001-01-15.nc"
    awap_data = xr.open_dataset(file_path_awap)
    lat = awap_data.lat.values
    lon = awap_data.lon.values
    
    base_path = "/scratch/iu60/xs5813/metric_results/"
    mymodel_path = f"{base_path}skil_dis/model_G_i000008_20240916-171624_with_huber/{year}/lead_time{leadingtime}_whole.npy"
    compare_path = ""

    if method == "QM":
        compare_path = f"/scratch/iu60/xs5813/qm/new_crps/save/crps_ss/QM/{year}/lead_time{leadingtime}_whole.npy"
    elif method == "climatology":
        #compare_path = f"/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/prob{pct}_climat_lead_time_{leadingtime}.npy"
        compare_path = f"/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/mean_climatology/{year}/window1/climat_lead_time_{leadingtime}.npy"
    elif method == "DESRGAN":
        compare_path = f"{base_path}skil/model_G_i000007_20240910-042620/{year}/lead_time{leadingtime}_whole.npy"
    else:
        print("输入的方法名称有误，请检查！")
        return

    # 加载数据
    my_model_data = np.load(mymodel_path).astype("float32")
    print("my_model_data",my_model_data)
    compare_data = np.load(compare_path).astype("float32")
    if compare_data.shape == (1, 413, 267):
        compare_data = np.squeeze(compare_data)

    # 绘图
    draw(my_model_data, compare_data, lon, lat, title=f"crps difference at leadingtime {leadingtime} compare with {method}")

def diff_average(year, leadingtime_start,leadingtime_end, method="QM"):
    # 获取纬度和经度信息
    file_path_awap = "/scratch/iu60/xs5813/Awap_data_bigger/2001-01-15.nc"
    awap_data = xr.open_dataset(file_path_awap)
    lat = awap_data.lat.values
    lon = awap_data.lon.values
    base_path = "/scratch/iu60/xs5813/metric_results/"
    model_prefix = f"{base_path}skil_dis/model_G_i000008_20240916-171624_with_huber/{year}/lead_time"
    compare_prefix = ""

    if method == "QM":
        compare_prefix = f"/scratch/iu60/xs5813/qm/new_crps/save/crps_ss/QM/{year}/lead_time"
    elif method == "climatology":
        compare_prefix = f"/scratch/iu60/xs5813/cli_metric_result/new_crps/save/climatology/mean_climatology/{year}/window1/climat_lead_time_"
    elif method == "DESRGAN":
        compare_prefix = f"{base_path}skil/model_G_i000007_20240910-042620/{year}/lead_time"
    else:
        print("输入的方法名称有误，请检查！")
        return

    # 读取并计算指定时间段的数据平均值
    my_model_avg = None
    compare_model_avg = None

    for leadingtime in range(leadingtime_start, leadingtime_end + 1):

        my_model_path = f"{model_prefix}{leadingtime}_whole.npy"
        compare_path = f"{compare_prefix}{leadingtime}_whole.npy"
        if method == "climatology":
            compare_path = f"{compare_prefix}{leadingtime}.npy"
        my_model_data = np.load(my_model_path).astype("float32")
        compare_data = np.load(compare_path).astype("float32")

        if compare_data.shape == (1, 413, 267):
            compare_data = np.squeeze(compare_data)

        if my_model_avg is None:
            my_model_avg = my_model_data
            compare_model_avg = compare_data
        else:
            my_model_avg += my_model_data
            compare_model_avg += compare_data
    #print("my_model_data",my_model_data)

    # 计算平均值
    print("my_model_avg",my_model_avg)
    print("compare_model_avg", compare_model_avg)
    my_model_avg /= (leadingtime_end - leadingtime_start + 1)
    compare_model_avg /= (leadingtime_end - leadingtime_start + 1)

    # 绘图
    draw(my_model_avg, compare_model_avg, lon, lat, title=f"Average crps difference at {year} from leadingtime {leadingtime_start} to {leadingtime_end} compared with {method}")
#diff(2018,1,method="climatology")
#diff(2018,0,method="QM")
#diff(2018,40,method="DESRGAN")
#diff_average(2007,0,41,method="climatology")
diff_average(2007,0,41,method="DESRGAN")
diff_average(2007,0,41,method="climatology")
diff_average(2007,0,41,method="QM")

diff_average(2006,0,41,method="DESRGAN")
diff_average(2006,0,41,method="climatology")
diff_average(2006,0,41,method="QM")

diff_average(2018,0,41,method="DESRGAN")
diff_average(2018,0,41,method="climatology")
diff_average(2018,0,41,method="QM")