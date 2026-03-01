🛰️ SCRAP: Satellite Collision Risk Assessment and Prediction

🚀 Overview

The exponential proliferation of space debris in Low Earth Orbit (LEO) has transformed collision avoidance into a critical daily operation. Current physics-based collision probability models often suffer from extreme false-positive rates due to dynamic orbital propagation uncertainties, leading to unnecessary and inefficient maneuver planning.

SCRAP (Satellite Collision Risk Assessment and Prediction) is a supervised machine learning framework designed to predict the final collision risk estimate of an encounter. Crucially, the system enforces a strict operational constraint: predictions are made using only telemetry data available at least two days prior to the Time of Closest Approach (TCA).

📊 Dataset

This project utilizes the European Space Agency's (ESA) Historical Conjunction Data Messages (CDMs) Database.

Size: 162,634 individual CDM records corresponding to 13,154 unique close-approach events.

Features: 103 numerical features encompassing:

Kinematics: Relative position and velocity vectors.

Uncertainty Matrices: 3D tracking radar covariance matrices.

Space Weather Indices: Solar radio flux (F10.7), Geomagnetic index (AP), and Wolf sunspot number (SSN).

Challenges: Extreme class imbalance. Over 98% of events possess a final risk value below the critical safety threshold ($10^{-6}$).

⚙️ Methodology & Pipeline

Operational Filtration (2-Day Cutoff): To simulate a realistic operational environment and avoid data leakage, all data occurring less than 2.0 days before TCA is dropped.

Time-Series Flattening: Variable-length time series are aggregated into fixed-length tabular inputs by calculating statistics like Last, Mean, Std, and Delta (Trend) for dynamic features.

Physics-Informed Feature Engineering:

Mahalanobis Distance ($D_M$): Normalizes spatial separation by the covariance matrix, measuring the distance relative to the radar's uncertainty ellipsoid.

Target Log-Transformation: The target risk variable is transformed into log-space ($y = \log_{10}(r + \epsilon)$) to stabilize variance across multiple orders of magnitude.

Custom Evaluation Metric: Standard MSE is inadequate for this highly imbalanced problem. The project optimizes a Custom Compound Loss Metric:

$$L = \frac{1}{F_2} \times MSE_{(r \ge 10^{-6})}$$

This explicitly assigns twice the weight to Recall over Precision ($F_2$-score), severely penalizing False Negatives (missed collisions), while minimizing error magnitude for high-risk predictions.

🧠 Machine Learning Models

Three models were benchmarked for this regression/classification hybrid task:

Random Forest Regressor: Robust baseline for comparison.

LightGBM: Computationally efficient leaf-wise gradient boosting.

XGBoost: Extreme gradient boosting utilizing the scale_pos_weight parameter to heavily penalize false negatives.

🏆 Key Results

Gradient boosting algorithms significantly outperformed traditional analytical baseline methods.

Best Model: XGBoost achieved the best overall performance with the lowest Compound Loss (0.224).

High-Risk Recall: XGBoost successfully correctly identified 98% of high-risk events, reducing the critical false-negative rate to just 2%.

Model Interpretability: SHAP (SHapley Additive exPlanations) analysis validated the importance of physics-informed feature engineering. The most influential predictive features were:

Latest physics-based risk estimate

Mahalanobis Distance

F10.7 Solar Flux (Space Weather)

Covariance Standard Deviation

📁 Repository Structure

├── data/                   # Raw and processed datasets (ESA CDMs)
├── notebooks/              # Jupyter notebooks for EDA and modeling
│   └── SCRAP_Satellite_Collision_Risk_Assessment_and_Prediction.ipynb
├── reports/                # Project proposal and final reports (PDF/LaTeX)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies


🛠️ How to Run

Clone the repository:

git clone [https://github.com/your-username/SCRAP-Collision-Prediction.git](https://github.com/your-username/SCRAP-Collision-Prediction.git)


Install the required dependencies:

pip install -r requirements.txt


Open the Jupyter Notebook to run the pipeline:

jupyter notebook notebooks/SCRAP_Satellite_Collision_Risk_Assessment_and_Prediction.ipynb


👥 Authors

Queen's University (CSAI 801, Winter 2026)

Mahmoud Alyosify

Mohamed Yahya

Mirna Embaby
