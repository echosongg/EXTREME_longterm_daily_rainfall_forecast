import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

def main(year, month, day, lat_lon_data):
    base_path = "/scratch/iu60/xs5813/"
    date_string = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    date_string2 = f"{year}{month.zfill(2)}{day.zfill(2)}"
    desrgan_modelname = "model_G_i000007_20240910-042620"
    prgan_modelname = "model_G_i000008_20240824-212330_with_huber"

    ensembles = [f"e0{i}" for i in range(1, 10)]  # e01 to e09

    for i, (lat, lon) in enumerate(zip(lat_lon_data['Lat'], lat_lon_data['Lon'])):
        results_list = []

        for ensemble in ensembles:
            for leadingtime in range(42):  # 0 to 41 leading times
                access_start_time = datetime.strptime(date_string, "%Y-%m-%d")
                access_end_time = access_start_time + timedelta(leadingtime)
                access_end_time_str = access_end_time.strftime("%Y-%m-%d")
                file_path_awap = f"{base_path}Awap_data_bigger/{access_end_time_str}.nc"
                
                # File paths for different models
                file_path_qm = f"{base_path}QM_cropped_data/{ensemble}/daq5_pr_{date_string2}_{ensemble}.nc"
                file_path_access = f"{base_path}Processed_data_bigger/{ensemble}/da_pr_{date_string2}_{ensemble}.nc"
                file_path_desrgan = f"{base_path}TestResults/DESRGAN/vTestRefactored/{desrgan_modelname}/{ensemble}/pr/{year}/{date_string}.nc"
                
                # Read datasets
                try:
                    awap_data = xr.open_dataset(file_path_awap).fillna(0)
                    access_data = xr.open_dataset(file_path_access).fillna(0)
                    qm_data = xr.open_dataset(file_path_qm, engine='netcdf4').fillna(0)
                    desrgan_data = xr.open_dataset(file_path_desrgan).fillna(0)

                    # Get the results for the specific lat/lon and leading time
                    awap_result = awap_data.sel(lat=lat, lon=lon, method='nearest').isel(time=0)['precip'].values
                    access_result = access_data.sel(lat=lat, lon=lon, method='nearest').isel(time=leadingtime)['pr'].values * 86400
                    qm_result = qm_data.sel(lat=lat, lon=lon, method='nearest').isel(time=leadingtime)['pr'].values
                    desrgan_result = desrgan_data.sel(lat=lat, lon=lon, method='nearest').isel(time=leadingtime)['pr'].values

                    # Append the results to the list, adding ensemble as a column
                    results_list.append({
                        'Lat': lat, 'Lon': lon, 'LeadingTime': leadingtime,
                        'Ensemble': ensemble,
                        'AWAP Result': awap_result, 'ACCESS Result': access_result,
                        'QM Result': qm_result, 'DESRGAN Result': desrgan_result
                    })

                except Exception as e:
                    print(f"Error processing lat={lat}, lon={lon}, ensemble={ensemble}, leadingtime={leadingtime}: {e}")
                    continue

        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)

        # Save the DataFrame to CSV for each lat/lon combination, all ensembles in one file
        file_name = f"results_{year}_{month}_{day}_lat_{lat}_lon_{lon}.csv"
        results_df.to_csv(file_name, index=False)
        print(f"Results saved to {file_name}")

# Example usage
lat_lon_data = {
    'Lat': [-35.2, -35.7, -27.2, -34.2, -33.3, -27.4, -30.3, -24.1, -37.3, 22, 25, 22, 24, 26, 29, 27, 29],
    'Lon': [147.5, 139.3, 149.1, 142.1, 149.1, 151.7, 115.5, 148.1, 143.0, 142, 143, 145, 144, 150, 153, 152, 152]
}

main(2007, "10", "01", lat_lon_data)
main(2006, "10", "01", lat_lon_data)
main(2018, "10", "01", lat_lon_data)
