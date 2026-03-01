#!/usr/bin/env python3
"""
IMPROVED SCRAP PIPELINE WITH ADVANCED FEATURES AND INTELLIGENT SCALING
========================================================================

This script runs the satellite collision risk prediction with:
1. Advanced time-series aggregation (trends, percentiles, volatility)
2. Enhanced physics feature extraction
3. Intelligent multi-scaler approach
4. Dual model training (XGBoost + LightGBM)
5. ESA challenge evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PART 1: DATA LOADING & PREPARATION  
# ============================================================================

def load_raw():
    """Load SCRAP dataset from Hugging Face."""
    ds = load_dataset("mahmoudalyosify/SCRAP", split="train")
    df = pd.DataFrame(ds)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

def filter_by_time(df, min_days=2.0):
    """Filter out CDMs not available at prediction time (< 2 days before TCA)."""
    return df[df["time_to_tca"] >= min_days].copy()

def log_transform(df):
    """Apply log transformation to skewed features (except 'risk' which is already in log10)."""
    skewed_cols = [
        "t_position_covariance_det", "c_position_covariance_det",
        "t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t", "t_sigma_n", "c_sigma_n",
    ]
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

# ============================================================================
# PART 2: ADVANCED AGGREGATION
# ============================================================================

def aggregate_event_advanced(df):
    """Advanced aggregation with trends, percentiles, and volatility."""
    target_col = "risk"
    numeric_cols = [
        "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
        "miss_distance", "relative_speed",
        "t_position_covariance_det", "c_position_covariance_det",
        "t_sigma_r", "c_sigma_r", "t_sigma_t", "c_sigma_t",
        "t_sigma_n", "c_sigma_n",
        "t_j2k_sma", "c_j2k_sma",
        "t_j2k_ecc", "c_j2k_ecc",
        "t_j2k_inc", "c_j2k_inc",
        "F10", "F3M", "SSN", "AP",
    ]

    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, True])

    def compute_advanced_stats(series):
        if len(series) < 2:
            series = pd.Series([series.iloc[0]] if len(series) == 1 else [0])
        
        stats = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'p25': series.quantile(0.25),
            'p50': series.quantile(0.50),
            'p75': series.quantile(0.75),
            'range': series.max() - series.min(),
            'cv': series.std() / (series.mean() + 1e-9),
            'skew': series.skew(),
            'trend': (series.iloc[-1] - series.iloc[0]) if len(series) > 1 else 0,
            'slope': np.polyfit(np.arange(len(series)), series, 1)[0] if len(series) > 1 else 0,
        }
        return stats
    
    event_agg_list = []
    for event_id, group in df.groupby("event_id"):
        row_dict = {"event_id": event_id}
        row_dict["risk"] = group[target_col].iloc[-1]
        
        for col in numeric_cols:
            if col in group.columns:
                series = group[col].dropna()
                if len(series) > 0:
                    stats = compute_advanced_stats(series)
                    for stat_name, stat_value in stats.items():
                        row_dict[f"{col}_{stat_name}"] = stat_value
        
        event_agg_list.append(row_dict)
    
    event_agg = pd.DataFrame(event_agg_list)
    logger.info(f"Advanced aggregation: {len(event_agg)} events × {len(event_agg.columns)} features")
    return event_agg

# ============================================================================
# PART 3: PHYSICS FEATURE ENGINEERING
# ============================================================================

def add_physics_features_advanced(df):
    """Add collision-risk physics indicators."""
    df = df.copy()
    
    # Speed-to-distance ratio
    df['speed_to_distance_mean'] = (
        df['relative_speed_mean'] / (df['miss_distance_mean'] + 1e-9)
    )
    df['speed_to_distance_max'] = (
        df['relative_speed_max'] / (df['miss_distance_min'] + 1e-9)
    )
    
    # Position uncertainty
    total_position_uncertainty = (
        df.get('t_sigma_r_mean', 0) + df.get('c_sigma_r_mean', 0) +
        df.get('t_sigma_t_mean', 0) + df.get('c_sigma_t_mean', 0) +
        df.get('t_sigma_n_mean', 0) + df.get('c_sigma_n_mean', 0)
    )
    df['uncertainty_to_distance'] = (
        total_position_uncertainty / (df['miss_distance_mean'] + 1e-9)
    )
    
    # Covariance determinant
    df['combined_cov_det_mean'] = (
        df.get('t_position_covariance_det_mean', 1e-10) *
        df.get('c_position_covariance_det_mean', 1e-10)
    ) ** 0.5
    
    # Covariance growth
    df['cov_growth_rate'] = (
        (df.get('t_position_covariance_det_max', 1e-10) - 
         df.get('t_position_covariance_det_min', 1e-10)) /
        (df.get('t_position_covariance_det_mean', 1e-10) + 1e-9)
    )
    
    # Orbital parameters
    delta_sma = np.abs(df.get('t_j2k_sma_mean', 0) - df.get('c_j2k_sma_mean', 0))
    delta_ecc = np.abs(df.get('t_j2k_ecc_mean', 0) - df.get('c_j2k_ecc_mean', 0))
    delta_inc = np.abs(df.get('t_j2k_inc_mean', 0) - df.get('c_j2k_inc_mean', 0))
    
    df['orbital_distance'] = (delta_sma / max(1e3, delta_sma.max() or 1) +
                              delta_ecc + delta_inc)
    
    # Solar activity interaction
    df['activity_interaction'] = (
        df.get('F10_mean', 0) * df['uncertainty_to_distance']
    )
    
    # Log-compress large values
    for col in ['combined_cov_det_mean', 'cov_growth_rate']:
        if col in df.columns:
            df[col] = np.log10(df[col] + 1)
    
    logger.info(f"Physics features added. Total: {df.shape[1]} features")
    return df

# ============================================================================
# PART 4: INTELLIGENT SCALING
# ============================================================================

class IntelligentScaler:
    """Multi-strategy scaler for different feature groups."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_groups = {}
    
    def fit(self, X):
        """Fit scalers to feature groups based on statistical properties."""
        feature_stats = {}
        
        for col in X.columns:
            if col in ['event_id', 'risk']:
                continue
            
            series = X[col].dropna()
            if len(series) == 0:
                continue
            
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
            outlier_ratio = outliers / len(series) if len(series) > 0 else 0
            skewness = series.skew()
            
            feature_stats[col] = {
                'outlier_ratio': outlier_ratio,
                'skewness': np.abs(skewness),
            }
        
        # Assign to scaler groups
        for col, stats in feature_stats.items():
            if stats['outlier_ratio'] > 0.05:
                group = 'robust'
            elif stats['skewness'] > 1.5:
                group = 'quantile'
            else:
                group = 'standard'
            
            self.feature_groups[col] = group
        
        # Create scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler(quantile_range=(10, 90))
        self.scalers['quantile'] = QuantileTransformer(n_quantiles=1000, 
                                                       output_distribution='uniform',
                                                       random_state=42)
        
        # Fit each
        for group_name in ['standard', 'robust', 'quantile']:
            group_cols = [col for col, grp in self.feature_groups.items() if grp == group_name]
            if group_cols:
                self.scalers[group_name].fit(X[group_cols])
        
        logger.info(f"IntelligentScaler fitted: {len(self.feature_groups)} features")
        return self
    
    def transform(self, X):
        """Apply appropriate scaler to each feature."""
        X_scaled = X.copy()
        
        for group_name in ['standard', 'robust', 'quantile']:
            group_cols = [col for col, grp in self.feature_groups.items() if grp == group_name]
            if group_cols and group_name in self.scalers:
                X_scaled[group_cols] = self.scalers[group_name].transform(X[group_cols])
        
        return X_scaled

# ============================================================================
# PART 5: ESA SCORING
# ============================================================================

def compute_esa_loss(y_true, y_pred, log_threshold=-6):
    """Compute official ESA challenge loss."""
    # Binary classification
    y_true_binary = (y_true >= log_threshold).astype(int)
    y_pred_binary = (y_pred >= log_threshold).astype(int)
    
    # Confusion matrix
    tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
    fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
    fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
    tn = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F2 score
    beta = 2
    beta_sq = beta ** 2
    if (beta_sq * precision + recall) == 0:
        f2 = 0
    else:
        f2 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    # MSE (high-risk only)
    high_risk_mask = y_true >= log_threshold
    if high_risk_mask.sum() > 0:
        mse_high = np.mean((y_true[high_risk_mask] - y_pred[high_risk_mask]) ** 2)
    else:
        mse_high = 0
    
    # Loss
    if f2 > 0:
        loss = (1.0 / f2) * mse_high
    else:
        loss = float('inf')
    
    return loss, f2, mse_high, precision, recall, int(tp), int(fp), int(fn)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("IMPROVED SCRAP PIPELINE WITH ADVANCED FEATURES")
    logger.info("="*80 + "\n")
    
    # Step 1: Load & Preprocess
    logger.info("[1/5] Loading and filtering data...")
    raw_df = load_raw()
    filtered_df = filter_by_time(raw_df)
    filtered_df = log_transform(filtered_df)
    logger.info(f"Loaded {len(filtered_df)} observations")
    
    # Step 2: Advanced aggregation
    logger.info("\n[2/5] Advanced event aggregation...")
    event_df = aggregate_event_advanced(filtered_df)
    
    # Step 3: Physics features
    logger.info("\n[3/5] Adding physics features...")
    event_df = add_physics_features_advanced(event_df)
    
    # Step 4: Handle missing values
    logger.info("\n[4/5] Data validation...")
    event_df = event_df.fillna(event_df.mean(numeric_only=True))
    event_df = event_df.replace([np.inf, -np.inf], np.nan)
    event_df = event_df.fillna(0)
    logger.info(f"Final data: {event_df.shape}")
    
    # Step 5: Train-test split with intelligent scaling
    logger.info("\n[5/5] Train-test split and scaling...")
    
    risk_threshold = -6
    y_binary = (event_df['risk'] >= risk_threshold).astype(int)
    
    feature_cols = [col for col in event_df.columns if col not in ['risk', 'event_id']]
    X = event_df[feature_cols].copy()
    y = event_df['risk'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Intelligent scaling
    scaler = IntelligentScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    # Train models
    logger.info("\n[6/5] Training models...")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        max_depth=7,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
        verbose=0
    )
    xgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)],
                  early_stopping_rounds=50,
                  verbose=False)
    
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        max_depth=7,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)],
                  early_stopping_rounds=50,
                  callbacks=[lgb.log_evaluation(period=0)])
    
    y_pred_lgb = lgb_model.predict(X_test_scaled)
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("ESA CHALLENGE EVALUATION (Advanced Pipeline)")
    logger.info("="*80)
    
    results = []
    for model_name, y_pred in [("XGBoost", y_pred_xgb), ("LightGBM", y_pred_lgb)]:
        loss, f2, mse_high, prec, recall, tp, fp, fn = compute_esa_loss(y_test.values, y_pred)
        
        mse_all = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  ESA Loss: {loss:.4f}")
        logger.info(f"  F2-Score: {f2:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  MSE (high-risk): {mse_high:.6e}")
        logger.info(f"  MSE (all): {mse_all:.6e}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        
        results.append({
            'Model': model_name,
            'ESA_Loss': loss,
            'F2_Score': f2,
            'Precision': prec,
            'Recall': recall,
            'MSE_HighRisk': mse_high,
            'MSE_All': mse_all,
            'R2': r2,
            'TP': tp,
            'FP': fp,
            'FN': fn
        })
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "esa_scores_improved.csv", index=False)
    
    with open(output_dir / "models_improved.pkl", "wb") as f:
        pickle.dump({'xgb': xgb_model, 'lgb': lgb_model, 'scaler': scaler}, f)
    
    logger.info(f"\n✓ Results saved to {output_dir}/esa_scores_improved.csv")
    logger.info(f"✓ Models saved to {output_dir}/models_improved.pkl")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()
