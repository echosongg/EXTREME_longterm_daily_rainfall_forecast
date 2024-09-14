import pandas as pd
import matplotlib.pyplot as plt
import os

def draw(year, metric="Brier", pct="99"):
    base_file = "/home/599/xs5813/EXTREME/crps_calculation_code/csv_files"
    save_path = "/home/599/xs5813/EXTREME/visual/line_photo"
    os.makedirs(save_path, exist_ok=True)  # Ensures the directory exists

    qm_base = f"{base_file}/model_G_i000006_20240401-042157/{year}/QM_{year}_window1.csv"
    climatology_base = f"{base_file}/model_G_i000006_20240401-042157/{year}/climatology_{year}_window1.csv"
    my_model_base = f"{base_file}/model_G_i000005_20240403-035316_with_huber/{year}/crps_ss_{year}_window1.csv"
    desrgan_base = f"{base_file}/model_G_i000006_20240401-042157/{year}/crps_ss_{year}_window1.csv"

    # Initialize empty lists
    qm = []
    climatology = []
    desrgan = []
    my_model = []

    qm = pd.read_csv(qm_base)[f"{metric}{pct}"]
    climatology = pd.read_csv(climatology_base)[f"{metric}{pct}"]
    my_model = pd.read_csv(my_model_base)[f"{metric}{pct}"]
    desrgan = pd.read_csv(desrgan_base)[f"{metric}{pct}"]

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each model's data with color-blind friendly colors
    plt.plot(qm, label='QM', linestyle='-', color='#377eb8')  # Blue
    plt.plot(climatology, label='Climatology', linestyle='--', color='#ff7f00')  # Orange
    plt.plot(desrgan, label='DESRGAN', linestyle='-.', color='#ffff33')  # Yellow
    plt.plot(my_model, label='ProbRainGAN', linestyle=':', color='#a65628')  # Bluish-gray

    title = f"{metric} {pct}% Comparison: Metric over lead times from 0 to 41 days for 72 SCFs made in {year}"
    # Adding titles and labels
    plt.title(title)
    plt.xlabel('Leading Time (1-42)')
    plt.ylabel(f'{metric} Score')
    plt.legend()

    # Save the plot instead of showing it
    plt.savefig(f"{save_path}/{metric}_{year}_{pct}_comparison.png")
    plt.close()

# Example of usage:
draw(2007)
