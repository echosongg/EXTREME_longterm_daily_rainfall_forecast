# 🌧️ Rainfall Prediction Project

This repository contains a full pipeline for rainfall prediction using deep learning models and statistical baselines. The workflow is divided into two main modules:

1. **Preprocessing**
2. **Model Training, Inference, and Evaluation**

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

This prepares inputs for bias correction using the QM method in the evaluation phase.

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

## 📊 Metrics Explanation

| Metric       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **Brier**    | Measures the accuracy of probabilistic forecasts at thresholds (0.95, etc.) |
| **CRPS**     | Compares the full distribution forecast with observations                   |
| **MAE**      | Median Absolute Error of the rainfall predictions                           |
| **Rel. Bias**| Relative bias of the predicted rainfall                                     |

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
│   ├── eval_distribution.py
│   └── eval_PEFGAN.py
├── crps_calculation_code/
│   ├── clim_table_csv.py
│   ├── climatology.py
│   └── evalQM.py
├── train.py  # for DESRGAN
```

---

For any issues or bugs, please raise an issue or contact the authors. Happy forecasting! 
