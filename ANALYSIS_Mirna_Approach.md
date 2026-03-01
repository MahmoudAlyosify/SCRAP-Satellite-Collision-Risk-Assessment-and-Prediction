# Technical Analysis: Mirna's Approach vs. ESA Challenge Requirements

**Analysis Date**: March 1, 2026  
**Analyst Role**: Principal ML Engineer & Data Architect  
**Report Level**: CRITICAL FINDINGS - Fundamental Approach Misalignment

---

## EXECUTIVE SUMMARY

**Verdict**: ⚠️ **INCORRECT APPROACH FOR THIS CHALLENGE**

Mirna achieved:
- F2 Score: **0.9998659937715145**
- MSE: **0.013400035733428623**
- Custom Loss: **0.013401831662344491**

**These scores are not valid for the ESA Collision Avoidance Challenge.** The approach violates fundamental project requirements.

---

## 1. PROBLEM FORMULATION MISMATCH

### What the ESA Challenge Requires:
- **Problem Type**: Continuous Regression
- **Target Variable**: Collision risk probability (in log₁₀ scale)
- **Value Range**: log₁₀(risk) ∈ [-30, -2.17] (continuous)
- **Prediction Output**: Continuous risk values r̂, not discrete classes

### What Mirna Implemented:
- **Problem Type**: Multi-class Classification (3 classes)
- **Target Variable**: Binned categories
  - 'High': risk ≤ -20
  - 'Medium': -20 < risk ≤ -10
  - 'Low': risk > -10
- **Prediction Output**: Discrete class labels

**CRITICAL ISSUE**: Mirna solved a **different problem** than the ESA challenge.

---

## 2. METRIC CALCULATION ERRORS

### Official ESA Challenge Metric:
```
L(r, r̂) = (1/F2) × MSE(r, r̂)

Where:
- F2 = binary F-β score (β=2) computed ONLY at threshold r ≥ 1e-6 (-6 in log₁₀)
- MSE computed ONLY for high-risk events (r ≥ 1e-6)
- High-risk class: r ≥ 1e-6
- Low-risk class: r < 1e-6
```

### What Mirna Calculated:
```python
# Mirna's implementation:
custom_loss = mse / f2

Where:
- mse = MSE across ENTIRE dataset using arbitrary mappings:
  - 'Low' → -20
  - 'Medium' → -15
  - 'High' → -5
- f2 = Weighted F2 across 3-class classification (NOT binary)
```

**THREE DISTINCT PROBLEMS**:

| Aspect | ESA Official | Mirna's Code | Issue |
|--------|-------------|--------------|-------|
| Target | Continuous values | 3 discrete classes | Different problem space |
| Threshold | 1e-6 = -6 (log₁₀) | Arbitrary: -20, -10 | Arbitrary binning |
| F2 Computation | Binary (high/low) | 3-class weighted | Wrong averaging method |
| MSE | High-risk only (r ≥ -6) | All predictions | Wrong dataset subset |
| y_pred Mapping | Continuous predictions | Class→Numeric mapping | Loss of information |

---

## 3. SUSPICIOUSLY HIGH PERFORMANCE

### Baseline (Correct Implementation):
- **XGBoost**: F2 = 0.2525, Loss = 57.87
- **LightGBM**: F2 = 0.2530, Loss = 59.06
- **Challenge Difficulty**: High-risk recall only 21.8% (missing 78% of collisions)

### Mirna's Results:
- **F2 = 0.9998659937715145** (99.98% accuracy in 3-class classification)
- **Custom Loss = 0.0134** (essentially zero error)

### Why This is Unrealistic:

1. **Class Imbalance Not Addressed**:
   - Random Forest with 100 trees tends to overfit on imbalanced data
   - No SMOTE, class weights, or stratification for collision risk levels

2. **Arbitrary Mapping Artifacts**:
   ```python
   bin_to_numeric = {'Low': -20, 'Medium': -15, 'High': -5}
   y_test_numeric = y_test.map(bin_to_numeric)  # ← Information loss
   ```
   - Converting 3 classes to 3 numeric values creates **zero variance** predictions
   - Any correct classification → near-perfect MSE by design

3. **Division by F2 Distortion**:
   - MSE/F2 is **not** the ESA formula: L = (1/F2) × MSE
   - These are mathematically different: 0.0134/0.9998 ≠ (1/0.9998) × 0.0134
   - Even if they were equal, dividing by near-1.0 doesn't penalize prediction error

---

## 4. FUNDAMENTAL ARCHITECTURAL ISSUES

### Issue 1: Random Forest Classifier (Regression Problem)
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

**Why This is Wrong**:
- ESA challenge is **continuous regression**, not classification
- RandomForestRegressor needed, not Classifier
- Classification assumes discrete decision boundaries; regression captures continuous risk

---

### Issue 2: Arbitrary Binning Destroys Information
```python
data['risk_bin'] = pd.cut(data['risk'], 
                          bins=[-np.inf, -20, -10, np.inf], 
                          labels=['High','Medium','Low'])
```

**Collateral Damage**:
- Original dataset: 162,634 observations with continuous risk values
- After binning: Loss of all within-class granularity
- Event at risk = -19.9 treated identically to risk = -20.1
- **Cannot recover continuous predictions from discrete classification**

---

### Issue 3: Wrong Risk Thresholds
```
Mirna's thresholds: -20, -10
ESA Challenge:     -6 (1e-6 in linear scale)
```

**Impact**:
- ESA defines high-risk as: log₁₀(P_collision) ≥ -6 (P ≥ 1e-6 = 0.0001%)
- Mirna defines high-risk as: risk ≤ -20 (P ≤ 1e-20 = 10^-20)
- **Mirna's "High" class is 10^14× more severe than ESA's high-risk threshold**
- Different class distributions, different evaluation semantics

---

### Issue 4: No Continuous Prediction Capability
```python
y_pred = rf.predict(X_test)  # Returns class labels: ['Low', 'Medium', 'High']
```

**The ESA Challenge Requires**:
- Continuous risk predictions: r̂ ∈ [-30, -2]
- Probability estimates or intermediate values
- This notebook can **never** produce ESA-valid submissions

---

## 5. DATA SCIENCE RED FLAGS

### Flag 1: Suspicious Perfect Stratification
```
Balanced Accuracy ≈ 1.0 on 3-class classification
F2 = 0.9998659937715145
```
- Indicates extreme overfitting or data leakage
- 100 trees on multi-class is insufficient for such performance
- Suggests class imbalance completely favors predictions

### Flag 2: Arbitrary Numeric Mapping
```python
{'Low': -20, 'Medium': -15, 'High': -5}
```
- Why these values? No justification
- MSE computed on arbitrary integer differences, not actual risk values
- Makes metric meaningless for collision risk assessment

### Flag 3: Incomplete Preprocessing
```python
X_train.fillna(X_train.median(), inplace=True)     # ← Basic imputation
X_test.fillna(X_train.median(), inplace=True)      # ← No scaling
X_train = X_train.clip(-1e6, 1e6)                  # ← Crude clipping
```
- No StandardScaler or RobustScaler
- No feature engineering (trends, percentiles, physics features)
- Competitive approaches use 300+ engineered features; this uses raw ~200 features

---

## 6. COMPARISON TO CORRECT IMPLEMENTATION

| Component | Mirna's Approach | Correct Approach | Status |
|-----------|-----------------|-----------------|--------|
| **Problem Type** | Classification | Regression | ❌ Wrong |
| **Target** | 3-class labels | Continuous values | ❌ Wrong |
| **Model** | RandomForestClassifier | XGBoost/LightGBM Regression | ❌ Wrong |
| **Threshold** | -20, -10 (arbitrary) | -6 (1e-6 per ESA) | ❌ Wrong |
| **F2 Computation** | 3-class weighted | Binary high/low | ❌ Wrong |
| **MSE Scope** | All predictions | High-risk only | ❌ Wrong |
| **Features** | ~200 raw features | 326 engineered features | ❌ Suboptimal |
| **Scaling** | None | Intelligent multi-scaler | ❌ Suboptimal |
| **F2 Score Achieved** | 0.9998 (invalid) | 0.2525 (valid) | ⚠️ Incomparable |

---

## 7. WHY THE SCORES APPEAR GOOD (But Aren't Valid)

### The Mathematical Artifact:
```
When converting 3 discrete classes to 3 numeric values [-20, -15, -5]:
- All predictions within a class are identical
- MSE = 0 when predictions are correct
- Small MSE values are guaranteed by design, not by model quality
- F2 score inflated by 3-class imbalance, not by collision detection
```

### Real-World Impact:
- Mirna's model would **fail the ESA challenge** because:
  1. Cannot submit class labels (needs continuous predictions)
  2. Predictions not in required [-30, -2] range  
  3. Metric computed on wrong data subset
  4. Evaluation threshold (-6) not reflected in training

---

## 8. WHAT MIRNA SHOULD DO

### Correction Path 1: Respect the Challenge Requirements
```python
# ✅ CORRECT APPROACH:
from sklearn.ensemble import RandomForestRegressor  # Regressor, not Classifier

# Prepare data
X = df_filtered.select_dtypes(include=np.number)
y = df_filtered['risk'].values  # Continuous, not binned

# Train continuous regression
rf_reg = RandomForestRegressor(n_estimators=500, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)  # Continuous predictions

# Evaluate with ESA metric
log_threshold = -6  # 1e-6 in linear
y_true_binary = (y_test >= log_threshold).astype(int)
y_pred_binary = (y_pred >= log_threshold).astype(int)

# Compute F2 (binary)
f2 = fbeta_score(y_true_binary, y_pred_binary, beta=2)

# Compute MSE (high-risk only)
high_risk_mask = y_test >= log_threshold
mse_high = mean_squared_error(y_test[high_risk_mask], y_pred[high_risk_mask])

# ESA Loss
esa_loss = (1.0 / f2) * mse_high if f2 > 0 else float('inf')
```

### Correction Path 2: Implement Advanced Features
- Add time-series aggregation (trends, percentiles, volatility)
- Include physics features (speed-to-distance, covariance interactions)
- Use Intelligent Scaling (StandardScaler + RobustScaler + QuantileTransformer)
- Apply class weighting or SMOTE for high-risk minority class

### Correction Path 3: Use Appropriate Models
- XGBoost Regressor (proven on this data)
- LightGBM Regressor with early stopping
- Ensemble methods with stratified validation

---

## 9. PROFESSIONAL ASSESSMENT

### Strengths in Mirna's Work:
✅ Proper temporal filtering (2-day cutoff)  
✅ Missing value handling  
✅ Data distribution analysis  
✅ Skewness detection and transformation  
✅ Train-test stratification  

### Critical Deficiencies:
❌ Fundamental problem misformulation (classification ≠ regression)  
❌ Incorrect metric calculation (not ESA compliant)  
❌ Arbitrary design choices (bin edges, numeric mapping)  
❌ Invalid score interpretation (cannot be compared to ESA baseline)  
❌ No continuous prediction capability  
❌ Insufficient feature engineering  
❌ Inadequate model selection for regression task  

### Confidence Assessment:
- **Probability that Mirna's scores are ESA-challenge-compliant**: **0%**
- **Probability of code passing ESA evaluation**: **0%**
- **Likelihood of genuine improvement over baseline**: Unknown (cannot assess)

---

## 10. RECOMMENDATIONS

**Immediate Actions** (Priority: CRITICAL):
1. Reframe problem as **continuous regression** task
2. Use **RandomForestRegressor** or XGBoost/LightGBM
3. Adopt **ESA-compliant evaluation**:
   - Binary classification only (high-risk ≥ 1e-6)
   - MSE computed on high-risk subset
   - F2 score with β=2
   - Loss formula: L = (1/F2) × MSE
4. Apply **correct risk threshold** (-6 in log₁₀ scale)

**Medium-term Optimization** (Priority: HIGH):
1. Implement **advanced feature engineering**:
   - Time-series aggregation (means, percentiles, trends)
   - Physics-based indicators (speed-to-distance, orbital deltas)
   - Uncertainty metrics (covariance growth)
2. Apply **intelligent scaling** (multiple scalers for heterogeneous features)
3. Deploy **ensemble methods** with proper cross-validation
4. Use **class weighting** or SMOTE for collision risk imbalance

**Validation Protocol** (Priority: ESSENTIAL):
- Validate against the **baseline scores** (F2 ≈ 0.25, Loss ≈ 58)
- Ensure predictions are **continuous** in [-30, -2] range
- Confirm **high-risk detection rate** (target: >50% recall)
- Compare against **XGBoost baseline** (current best: Loss = 57.87)

---

## CONCLUSION

Mirna's approach demonstrates good foundational data science skills (EDA, preprocessing, visualization) but **fundamentally misunderstands the ESA Collision Avoidance Challenge requirements**.

The reported scores (F2 = 0.9998, Loss = 0.0134) are **mathematically invalid** for this challenge because:
1. **Problem type mismatch**: Classification ≠ Regression
2. **Metric violations**: Wrong threshold, wrong F2, wrong MSE scope
3. **Prediction incompatibility**: Cannot produce required continuous outputs
4. **Evaluation misalignment**: Results not comparable to ESA standards

**Recommendation**: Restart with proper problem formulation while preserving good preprocessing practices.

---

**Prepared by**: Principal ML Engineer, Data Architecture Division  
**Classification**: TECHNICAL ANALYSIS  
**Status**: Final Assessment  
