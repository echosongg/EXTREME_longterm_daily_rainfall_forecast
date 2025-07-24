# 🌧️

This repository contains a full pipeline for rainfall prediction using deep learning models and statistical baselines. The workflow is divided into three main modules:

1. **Preprocessing & Metric Calculation**
2. **Model Training, Inference, and Evaluation**
3. **Results Output and Visualization**

---

## Module 1: Preprocessing & Metric Calculation

This module prepares input data and computes shared evaluation metrics (e.g., alpha).

---

### **1. AGCD Ground Truth (5km resolution)**

#### ➤ Data Processing
```bash
python preprocessing/Data_process_code/agcd_mask_processing.py
```

---

### **2. ACCESS-S2 Forecast Data (60km → Interpolated to 40km)**

#### ➤ Data Processing
```bash
python preprocessing/Data_process_code/ACCESS_e01.py
```

---

### **3. Quantile Mapping (QM) Preprocessing**

#### ➤ Data Preparation
```bash
python preprocessing/QM_pre/QM_data_crop_e1.py
```

---

---

## 🚀 Module 2: Models and Evaluation

This module runs deep learning models, baseline evaluations, and computes predictive performance metrics.

---

### 🌧️ 1. PRGAN (Probabilistic Rainfall GAN)

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

When `generate = False`: outputs **rainfall probability (p)** and **gamma distribution parameters**  
When `generate = True`: generates rainfall values via:

Mathematically:

```math
\hat{R} = \mathbf{1}_{\{p > 0.5\}} \cdot (\alpha \cdot \beta)
```

#### 4. Distribution-based evaluation:
```bash
python eval_distribution.py
```

#### 5. Relative bias evaluation:
```bash
python model_built/eval_PEFGAN.py
```

#### 6. Alpha metric evaluation:
```bash
python model_built/eval_alpha_dis.py
```

---

### ⚡ 2. DESRGAN

#### 1. Train the model:
```bash
python train.py
```

#### 2. Run predictions:
```bash
python model_built/test.py
```

#### 3. Evaluate standard metrics:
```bash
python model_built/eval_PEFGAN.py
```

Generates:
- Brier scores (0.95, 0.99, 0.995)
- CRPS
- MAE (Median)
- Relative Bias

#### 4. Alpha metric evaluation:
```bash
python model_built/eval_alpha.py
```

---

### 🌦️ 3. Climatology Baseline

#### 1. Create climatology lookup tables:
```bash
python crps_calculation_code/clim_table_csv.py
```

#### 2. Run climatology forecast & metrics:
```bash
python crps_calculation_code/climatology.py
```

#### 3. Alpha metric evaluation:
```bash
python crps_calculation_code/climatology_alpha.py
```

---

### 📉 4. Quantile Mapping (QM)

#### 1. Evaluate QM forecast:
```bash
python crps_calculation_code/evalQM.py
```

#### 2. Alpha metric evaluation:
```bash
python crps_calculation_code/qm_alpha.py
```

---

## 🗂️ Module 3: Results Output and Visualization

This module exports evaluation results and visualizes spatial metrics.

---

### **1. Output Average Metrics Over 41 Lead Times**

```bash
python crps_calculation_code/crps_ss.py
```

Generates CSV tables with:
- CRPS, MAE, Brier scores, alpha, 7 metrics
- Per-lead-time average metrics

---

### **2. Visualize Spatial Metric Differences**

#### a. Brier Score Difference Maps
```bash
python visual/diff_brier.py
```

#### b. CRPS Difference Maps
```bash
python visual/diff_crps.py
```

These generate interpolated spatial maps of metric differences across regions.

---

## 📊 Metrics Summary

| Metric       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **Brier**    | Accuracy of probabilistic forecasts at thresholds (e.g. 0.95)              |
| **CRPS**     | Continuous Ranked Probability Score: full distribution accuracy            |
| **MAE**      | Median Absolute Error of rainfall values                                   |
| **Rel. Bias**| Bias between predicted vs observed rainfall                                |
| **Alpha**    | Characterizes predicted distribution sharpness/uncertainty                 |

---

## 🔧 Requirements

Please ensure the following dependencies are installed:
- Python >= 3.7
- PyTorch >= 1.8
- NumPy, Pandas, Matplotlib
- xarray, netCDF4, Basemap (for spatial plotting)
- Other packages from `requirements.txt`

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
│   ├── eval_PEFGAN.py
│   ├── eval_distribution.py
│   ├── eval_alpha_dis.py
│   └── eval_alpha.py
├── crps_calculation_code/
│   ├── clim_table_csv.py
│   ├── climatology.py
│   ├── climatology_alpha.py
│   ├── evalQM.py
│   ├── qm_alpha.py
│   └── crps_ss.py
├── visual/
│   ├── diff_brier.py
│   └── diff_crps.py
```

---

For questions or issues, please open an issue on the repository.  
Happy forecasting! 🌧️📈
