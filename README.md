# Rainfall Prediction Models - PRGAN, DESRGAN, Climatology, and QM

This repository contains implementations for various rainfall prediction approaches including PRGAN, DESRGAN, Climatology, and Quantile Mapping (QM). The models predict rainfall probability and distribution-based metrics over a sequence of timesteps.

---

## 1. PRGAN (My Model)

### **Step-by-step Usage**

#### 1. Train the initial PRGAN model:
```bash
python model_built/train.py
```

This will save a pretrained model in the output directory.

#### 2. Fine-tune the model using:
```bash
python model_built/pretrain.py
```

Make sure the pretrained model is correctly loaded inside `pretrain.py`.

#### 3. Predict rainfall probability and distribution parameters:
```bash
python model_built/test_pab.py
```

- When `generate = False`: outputs the predicted **rainfall probability (p)** and **gamma distribution parameters**:  
  - \( \alpha \) (shape),  
  - \( \beta \) (scale)

- When `generate = True`: generates rainfall values using the following formula:

### **Rainfall Generation Equation**

The rainfall value \( \hat{R} \) is computed as:

```math
\hat{R} = \mathbf{1}_{\{p > 0.5\}} \cdot (\alpha \cdot \beta)
```

#### 4. Evaluate distribution-based metrics:
```bash
python model_built/eval_distribution.py
```

This will generate rainfall forecasts for 41 lead times and compute:
- Brier score at thresholds: 0.95, 0.99, 0.995
- CRPS (Continuous Ranked Probability Score)
- MAE (Median of 9 esemble members)

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

```

---

For any issues or bugs, please raise an issue or contact the authors. Happy forecasting! 
