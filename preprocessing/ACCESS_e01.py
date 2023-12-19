import xarray as xr
import os
import numpy as np
import cv2
import sys

from mpl_toolkits.basemap import maskoceans
e_list = ['e01']
for e_num in e_list:
    for file in (os.listdir("/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/" + e_num + "/")):
        # if int(file[6:10]) < 1990 or int(file[6:10]) > 2001:
        #     continue
        if int(file[6:10]) not in [2018, 2012, 2010, 2007, 2009]:
            continue
        ds_raw = xr.open_dataset("/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/" + e_num + "/" + file)
        ds_raw = ds_raw.fillna(0)
        da_selected = ds_raw.isel(time=0)['pr']
        # crop the Australian territory
        lon = ds_raw["lon"].values
        lat = ds_raw["lat"].values
        #change it to queensland
        a = np.logical_and(lon >= 140.6, lon <= 153.9)
        b = np.logical_and(lat >= -39.2, lat <= -18.6)

        da_selected_au = da_selected[b, :][:, a].copy()

        # resize lat & lon
        n = 1.5  # 60km -> 30km   == 2.0 scale

        # linspace is used to chop the interval into equally distributed value like np.linspace(3, 5, 3) = [3, 4, 5]
        # enlarge the original image to 12 times
        # Explanation[1]
        size = (int(da_selected_au.lon.size * n),
                int(da_selected_au.lat.size * n))
        new_lon = np.linspace(
            da_selected_au.lon[0], da_selected_au.lon[-1], size[0])
        new_lon = np.float32(new_lon)
        new_lat = np.linspace(da_selected_au.lat[0], da_selected_au.lat[-1], size[1])
        new_lat = np.float32(new_lat)
        # Explanation [2]
        lons, lats = np.meshgrid(new_lon, new_lat)

        # interp and merge
        i = ds_raw['time'].values[0]
        da_selected = ds_raw.sel(time=i)['pr']
        da_selected_au = da_selected[b, :][:, a].copy()
        temp = cv2.resize(da_selected_au.values, size, interpolation=cv2.INTER_CUBIC)
        temp = np.clip(temp, 0, None)
        # mask
        temp = maskoceans(lons, lats, temp, resolution='c', grid=1.25)

        da_interp = xr.DataArray(temp, dims=("lat", "lon"), coords={"lat": new_lat, "lon": new_lon, "time": i}, name='pr')
        ds_total = xr.concat([da_interp], "time")

        for i in ds_raw['time'].values[:60]:
            ds_selected_domained = ds_raw.sel(time=i)['pr']
            da_selected_au = ds_selected_domained[b, :][:, a].copy()
            temp = cv2.resize(da_selected_au.values, size, interpolation=cv2.INTER_CUBIC)
            temp = np.clip(temp, 0, None)
            temp = maskoceans(lons, lats, temp, resolution='c', grid=1.25)
            da_interp = xr.DataArray(temp, dims=("lat", "lon"), coords={"lat": new_lat, "lon": new_lon, "time": i},
                                     name='pr')
            expanded_da = xr.concat([da_interp], "time")
            ds_total = xr.merge([ds_total, expanded_da])

        # save to netcdf
        # Explanation [3]
        #ds_total.to_netcdf("/scratch/iu60/xs5813/Processed_data/" + e_num + "/" + file)
        #ds_raw.close()
        try:
            ds_total.to_netcdf("/scratch/iu60/xs5813/Pqrocessed_data/" + e_num + "/" + file)
            ds_raw.close()
        except Exception as e:
            print(f"保存文件时出现错误：{e}")

        