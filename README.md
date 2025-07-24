# 🌧️ Rainfall Prediction Project

This repository contains a full pipeline for rainfall prediction using deep learning models and statistical baselines. The workflow is divided into three main modules:

1. **Preprocessing**
2. **Model Training, Inference, and Evaluation**
3. **Results Output and Visualization**

---

## 📦 Module 1: Preprocessing

This module prepares both observation and model data as input for the learning and evaluation phases.

### **1. AGCD Ground Truth (5km resolution)**

Run the following script to process the AGCD rainfall ground truth data:

```bash
python preprocessing/Data_process_code/agcd_mask_processing.py
```

This step masks and aligns the high-resolution (5km) AGCD dataset for supervised training and evaluation.

---

### **2. ACCESS-S2 Forecast Data (60km → Interpolated to 40km)**

Run the following script to process the ACCESS-S2 model forecast data:

```bash
python preprocessing/Data_process_code/ACCESS_e01.py
```

This script:
- Reads original ACCESS-S2 data at 60km resolution
- Interpolates the data to 40km resolution for compatibility with the learning models

---

### **3. Quantile Mapping (QM) Preprocessing**

Crop and prepare the quantile mapping input data by running:

```bash
python preprocessing/QM_pre/QM_data_crop_e1.py
```

---

## 🚀 Module 2: Models and Evaluation

This module runs deep learning models, baseline evaluations, and metrics computation.

---

## 🌧️ 1. PRGAN (Probabilistic Rainfall GAN)

### **Step-by-step Usage**

#### 1. Train the initial PRGAN model:
```bash
python model_built/train.py
```

#### 2. Fine-tune the model using:
```bash
python model_built/pretrain.py
```

#### 3. Predict rainfall probability and distribution parameters:
```bash
python model_built/test_pab.py
```

- When `generate = False`: outputs the predicted **rainfall probability (p)** and **gamma distribution parameters**:  
  - \( \alpha \) (shape),  
  - \( \beta \) (scale)

- When `generate = True`: generates rainfall values using the following formula:

### **Rainfall Generation Equation**

```python
def generate_sample(bg_output):
    p_pred = torch.sigmoid(bg_output[:, 0, :, :]).unsqueeze(1)  # rain probability
    p_pred = (p_pred > 0.5).float()
    alpha_pred = torch.exp(bg_output[:, 1, :, :]).unsqueeze(1)  # shape parameter
    beta_pred = torch.exp(bg_output[:, 2, :, :]).unsqueeze(1)   # scale parameter
    return p_pred * (alpha_pred * beta_pred)
```

Mathematically:

```math
\hat{R} = \mathbf{1}_{\{p > 0.5\}} \cdot (\alpha \cdot \beta)
```

#### 4. Evaluate distribution-based metrics:
```bash
python eval_distribution.py
```

#### 5. Evaluate relative bias:
```bash
python model_built/eval_PEFGAN.py
```

---

## ⚡ 2. DESRGAN

### **Step-by-step Usage**

#### 1. Train the model:
```bash
python train.py
```

#### 2. Run predictions:
```bash
python model_built/test.py
```

#### 3. Evaluate all metrics including distribution and relative bias:
```bash
python model_built/eval_PEFGAN.py
```

This will generate rainfall forecasts for 41 lead times and compute:
- Brier scores (0.95, 0.99, 0.995)
- CRPS
- MAE (Median)
- Relative Bias

---

## 🌦️ 3. Climatology Baseline

### **Step-by-step Usage**

#### 1. Generate climatology tables:
```bash
python crps_calculation_code/clim_table_csv.py
```

#### 2. Run the climatology-based forecast and evaluation:
```bash
python crps_calculation_code/climatology.py
```

---

## 📉 4. Quantile Mapping (QM)

### **Step-by-step Usage**

#### 1. Run quantile-mapping-based evaluation:
```bash
python crps_calculation_code/evalQM.py
```

---

## 🗂️ Module 3: Results Output and Visualization

This module exports evaluation results and visualizes spatial metrics across the region and time.

---

### **1. Output CSV of Averaged Results (41 Lead Times)**

Run the following script to compute and save average metrics over all lead times:

```bash
python crps_calculation_code/crps_ss.py
```

This will generate CSV tables with average Brier scores, CRPS, MAE, etc.

---

### **2. Visualize Spatial Metric Differences**

#### a. Brier Score Difference Maps

```bash
python visual/diff_brier.py
```

This creates spatial visualizations showing Brier score differences between methods or lead times.

#### b. CRPS Difference Maps

```bash
python visual/diff_crps.py
```

This creates interpolated spatial maps for CRPS differences.

---

## 📊 Metrics Explanation

| Metric       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **Brier**    | Measures the accuracy of probabilistic forecasts at thresholds (0.95, etc.) |
| **CRPS**     | Compares the full distribution forecast with observations                   |
| **MAE**      | Median Absolute Error of the rainfall predictions                           |
| **Rel. Bias**| Relative bias of the predicted rainfall                                     |

---

## 🔧 Requirements

Please ensure the following dependencies are installed:
- Python >= 3.7
- PyTorch >= 1.8
- NumPy, Pandas, Matplotlib
- xarray, netCDF4, Basemap (for spatial plotting)
- Other requirements in `requirements.txt` (if present)

---

## 📁 Directory Structure

```
.
├── preprocessing/
│   ├── Data_process_code/
│   │   ├── agcd_mask_processing.py
│   │   └── ACCESS_e01.py
│   └── QM_pre/
│       └── QM_data_crop_e1.py
├── model_built/
│   ├── train.py
│   ├── pretrain.py
│   ├── test_pab.py
│   ├── test.py
│   └── eval_PEFGAN.py
├── crps_calculation_code/
│   ├── clim_table_csv.py
│   ├── climatology.py
│   ├── evalQM.py
│   └── crps_ss.py
├── visual/
│   ├── diff_brier.py
│   └── diff_crps.py
├── eval_distribution.py
├── train.py  # for DESRGAN
```

---

For any issues or bugs, please raise an issue or contact the authors. Happy forecasting! 🌧️📈
