
# 🛰️ SCRAP: Satellite Collision Risk Assessment and Prediction

## 🚀 Overview

The exponential growth of space debris in Low Earth Orbit (LEO) has made collision avoidance a critical operational challenge. Traditional physics-based collision probability models often generate extremely high false-positive rates due to orbital propagation uncertainties, leading to unnecessary and costly satellite maneuvers.

**SCRAP (Satellite Collision Risk Assessment and Prediction)** is a supervised machine learning framework that predicts the final collision risk estimate of a satellite encounter.

> ⚠️ Operational Constraint:  
All predictions are made using **only telemetry data available at least 2 days before the Time of Closest Approach (TCA)** — preventing data leakage and simulating real-world operational decision-making.

---

## 📊 Dataset

This project utilizes the European Space Agency (ESA) Historical Conjunction Data Messages (CDMs) Database.

- **Size:** 162,634 CDM records  
- **Events:** 13,154 unique close-approach events  
- **Features:** 103 numerical features  

### Feature Categories

- **Kinematics**
  - Relative position vectors
  - Relative velocity vectors  

- **Uncertainty Matrices**
  - 3D radar covariance matrices  

- **Space Weather Indices**
  - F10.7 Solar Radio Flux  
  - AP Geomagnetic Index  
  - Wolf Sunspot Number (SSN)

### Key Challenge

⚠️ **Extreme Class Imbalance**

- Over **98%** of events have final risk < `1e-6`
- Very few high-risk collision events

---

## ⚙️ Methodology & Pipeline

### 1️⃣ Operational Filtration (2-Day Cutoff)

All observations occurring less than **2.0 days before TCA** are removed to:

- Prevent data leakage  
- Simulate real operational constraints  

---

### 2️⃣ Time-Series Flattening

Variable-length CDM sequences are converted into fixed-length tabular features using:

- `Last`
- `Mean`
- `Standard Deviation`
- `Delta (Trend)`

---

### 3️⃣ Physics-Informed Feature Engineering

#### 🔹 Mahalanobis Distance ($D_M$)

Normalizes spatial separation using the covariance matrix:

D_M = sqrt((x - μ)^T Σ⁻¹ (x - μ))

Measures relative distance inside the radar uncertainty ellipsoid.

---

#### 🔹 Log-Transformation of Target

y = log10(r + ε)

This stabilizes variance across multiple orders of magnitude.

---

### 4️⃣ Custom Compound Loss Metric

Standard MSE is inadequate for highly imbalanced collision risk prediction.

Custom loss:

L = (1 / F2) × MSE (r ≥ 1e-6)

Where:

- F2 score weights **Recall twice as much as Precision**
- False Negatives (missed collisions) are heavily penalized
- Error magnitude for high-risk predictions is minimized

---

## 🧠 Machine Learning Models

| Model | Description |
|-------|------------|
| Random Forest | Strong baseline ensemble method |
| LightGBM | Efficient leaf-wise gradient boosting |
| XGBoost | Extreme gradient boosting with `scale_pos_weight` |

---

## 🏆 Key Results

### ✅ Best Model: XGBoost

- **Lowest Compound Loss:** `0.224`
- **High-Risk Recall:** `98%`
- **False-Negative Rate:** `2%`

Gradient boosting significantly outperformed traditional analytical baseline methods.

---

## 🔎 Model Interpretability

SHAP (SHapley Additive exPlanations) analysis confirmed the importance of physics-informed features.

### Most Influential Features

1. Latest physics-based risk estimate  
2. Mahalanobis Distance  
3. F10.7 Solar Flux  
4. Covariance Standard Deviation  

---

## 📁 Repository Structure

```

├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── SCRAP_Satellite_Collision_Risk_Assessment_and_Prediction.ipynb
│
├── reports/
│   ├── proposal.pdf
│   └── final_report.pdf
│
├── src/
├── requirements.txt
└── README.md

````

---

## 🛠️ Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/SCRAP-Collision-Prediction.git
cd SCRAP-Collision-Prediction
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the notebook

```bash
jupyter notebook notebooks/SCRAP_Satellite_Collision_Risk_Assessment_and_Prediction.ipynb
```

---

## 🧪 Tech Stack

* Python 3.10+
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📌 Future Work

* Real-time streaming collision risk prediction
* Temporal deep learning models (LSTM / Transformer)
* Integration with operational satellite maneuver planning systems
* Deployment as an API service

---

## 👥 Authors

**Queen’s University – CSAI 801 (Winter 2026)**

* Mahmoud Alyosify
* Mohamed Yahya
* Mirna Embaby


