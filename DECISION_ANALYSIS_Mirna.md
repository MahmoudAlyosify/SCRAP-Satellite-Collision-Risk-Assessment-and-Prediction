# DECISION ANALYSIS: Where Mirna's Approach Diverged from ESA Requirements

## DECISION TREE: What Went Wrong?

```
START: ESA Collision Avoidance Challenge
│
├─→ DECISION 1: "What type of problem is this?"
│   │
│   Mirna's Choice:  Classification (3-class)
│   ✗ WRONG → Risk levels binned into discrete categories
│   │         No continuous predictions possible
│   │         Results: F2 = 0.9998 (invalid)
│   │
│   Correct Choice: Regression (continuous)
│   ✓ RIGHT → Predict continuous collision probability log₁₀(P)
│            Can generate ESA-required outputs
│            Results: F2 ≈ 0.25 (realistic & competitive)
│
├─→ DECISION 2: "How should I define risk levels?"
│   │
│   Mirna's Choice:  Arbitrary bins: {≤-20, -20 to -10, >-10}
│   ✗ WRONG → Not aligned with ESA specification
│   │         Threshold -20 is 10^14 times more extreme than needed
│   │         Creates artificially imbalanced classification
│   │
│   Correct Choice: ESA's binary threshold: -6 (= log₁₀(1e-6))
│   ✓ RIGHT → Based on collision probability (0.0001%)
│            Aligns with challenge specification
│            Forces model to learn realistic risk boundaries
│
├─→ DECISION 3: "Which model type should I use?"
│   │
│   Mirna's Choice:  RandomForestClassifier
│   ✗ WRONG → Designed for discrete outputs (class labels)
│   │         Cannot produce continuous predictions
│   │         Output: ['Low', 'Medium', 'High']
│   │
│   Correct Choice: RandomForestRegressor (or XGBoost/LightGBM Regressor)
│   ✓ RIGHT → Designed for continuous outputs (real values)
│            Can produce predictions like -6.23, -4.78, etc.
│            Output: continuous array of risk values
│
├─→ DECISION 4: "How do I compute F2 score?"
│   │
│   Mirna's Choice:  fbeta_score(..., average='weighted')
│   ✗ WRONG → Computes 3-class weighted F2
│   │         Metric: F2_weighted = (1+4)*(P*R)/(4*P + R) per class
│   │         Result: F2 = 0.9998 (inflated by class imbalance)
│   │
│   Correct Choice: Binary fbeta_score on threshold classification
│   ✓ RIGHT → Define binary classes at ESA threshold (-6)
│            High-risk: y_pred >= -6
│            Low-risk: y_pred < -6
│            Compute F2 only on this binary task
│            Result: F2 ≈ 0.25 (realistic for hard problem)
│
├─→ DECISION 5: "Which data should I use for MSE?"
│   │
│   Mirna's Choice:  All predictions (entire test set)
│   ✗ WRONG → MSE = (1/n_total) * Σ(y_test - y_pred)²
│   │         Dominated by easy-to-predict low-risk events
│   │         Masks poor performance on critical collisions
│   │
│   Correct Choice: High-risk events only (r >= -6)
│   ✓ RIGHT → MSE = (1/n_high_risk) * Σ(y_high - ŷ_high)²
│            Penalizes collision prediction errors
│            Forces model to minimize high-impact mistakes
│
├─→ DECISION 6: "What's the loss formula?"
│   │
│   Mirna's Choice:  L = MSE / F2
│   ✗ WRONG → Not the official formula
│   │         Incorrect error metric
│   │         Ranges: (0.0134 / 0.9998) ≈ 0.0134
│   │
│   Correct Choice: L(r, r̂) = (1/F2) × MSE(r,r̂) [high-risk only]
│   ✓ RIGHT → Official ESA challenge formula
│            Correct penalty structure
│            Ranges: (1/0.25) × 14.6 ≈ 58.4
│
└─→ FINAL RESULT
    │
    Mirna:  F2 = 0.9998, Loss = 0.0134 (INVALID - cannot submit)
    ✗ Cannot compete in ESA challenge
    ✗ Submitted to wrong evaluation system
    ✗ Scores meaningless for competition
    │
    Correct: F2 = 0.25, Loss = 57.87 (VALID - ready to submit)
    ✓ Can submit continuous predictions
    ✓ Evaluated on official ESA metric
    ✓ Competitive with baseline models
```

---

## The Cascading Error Effect

Each wrong decision amplified the mistakes:

```
Decision 1 (Classification)
    ↓ (forces discrete outputs)
    ├→ Decision 3 (ClassifierModel)
    │  ↓ (can't produce continuous values)
    │  └→ Decision 6 (Wrong loss formula) [can only measure class accuracy]
    │
    └→ Decision 2 (Arbitrary bins)
       ↓ (creates imbalance)
       └→ Decision 4 (3-class F2)
          ↓ (inflates metric)
          └→ Final Result: F2 = 0.9998 (misleading)
```

---

## The Three Critical Junctures

### Critical Point 1: PROBLEM TYPE (Decision 1)
```
"Is this a CLASSIFICATION or REGRESSION problem?"

❌ Mirna said: "Classification! I'll bin the risk into 3 categories."
   → Locked into discrete outputs
   → No way to recover continuous predictions

✓ Correct answer: "Regression! Predict continuous risk values."
   → Can output any real number in [-30, -2]
   → Supports both classification and continuous evaluation
```

### Critical Point 2: METRIC DEFINITION (Decisions 4-6)
```
"How do I evaluate the model fairly?"

❌ Mirna said: "I'll compute F2 across 3 classes and divide MSE by F2."
   → F2 inflated to 0.9998 by class imbalance
   → MSE includes easy-to-predict cases
   → Not comparable to ESA baseline

✓ Correct answer: "I'll use binary F2 at threshold -6, MSE on high-risk only."
   → F2 ≈ 0.25 realistically reflects collision detection difficulty
   → MSE penalizes errors where it matters (collision events)
   → Directly comparable to XGBoost baseline (57.87)
```

### Critical Point 3: THRESHOLD SELECTION (Decision 2)
```
"Where should I draw the line between high-risk and low-risk?"

❌ Mirna said: "-20 and -10 seem reasonable."
   → Arbitrary, not based on problem physics
   → Creates 90%+ imbalance (most data in one class)
   → Misleading evaluation

✓ Correct answer: "-6 in log₁₀ scale (1e-6 probability threshold)"
   → Based on ESA specification
   → ~10% data in high-risk class (challenging but learnable)
   → Aligns with real collision risk thresholds
```

---

## Graphical Comparison

### Classification Approach (Mirna):
```
Risk Space (-30 to -2):
│
│  Low Class          Medium Class      High Class
│  (risk > -10)       (-20 to -10)      (risk < -20)
│  ████████░░░░░░     ████████░░░░░░    ████████░░░░░░
│  [most of data]     [some data]       [rare data]
│
Prediction Output:    ['Low', 'Low', 'Medium', 'Low', 'High']
                      ↑ discrete class labels only
                      ✗ Cannot express values like -6.23
```

### Regression Approach (Correct):
```
Risk Space (-30 to -2):
│
│  -30  -20  -10  -6   0
│   •••  ••••  ••••  ⚠️ ••••  •••••
│   ↓    ↓     ↓     ESA Threshold
│   model output: continuous predictions
│   [-28.4, -15.3, -6.8, -4.2, -3.7, ...]
│   ↑ can express any value in range
│   ✓ Can output specific collision risk
```

---

## Key Metrics Comparison

| Metric | Mirna's Result | Reality | Why Different |
|--------|---|---|---|
| **F2 Score** | 0.9998 | ≈ 0.25 | 3-class vs binary; class imbalance |
| **MSE** | 0.0134 | ≈ 14.6 | All data vs high-risk only |
| **Loss** | 0.0134 | ≈ 57.87 | Wrong formula & scope |
| **Interpretability** | "99.98% accuracy" | "22% collision detection" | Classification vs regression; threshold |
| **ESA Compliant** | ✗ No | ✓ Yes | Problem formulation mismatch |

---

## The F2 Score Mystery: Why 0.9998?

```python
# Mirna's 3-class calculation:
from sklearn.metrics import fbeta_score

y_test = ['High', 'Low', 'Low', 'Medium', 'High', 'Low', ...]  # 3 classes
y_pred = ['High', 'Low', 'Low', 'Medium', 'High', 'Low', ...]  # Perfect predictions

f2 = fbeta_score(y_test, y_pred, beta=2, average='weighted')
# Result: ≈ 0.9998 ✗ WRONG METRIC


# Correct binary calculation:
y_test_binary = (y_test_continuous >= -6).astype(int)     # [1,0,0,1,1,0,...]
y_pred_binary = (y_pred_continuous >= -6).astype(int)     # [0,0,0,1,1,0,...]

f2 = fbeta_score(y_test_binary, y_pred_binary, beta=2)
# Result: ≈ 0.25 ✓ CORRECT METRIC
# ↑ Only 0/4 high-risk collision events correctly predicted!
```

The F2 score OF 0.9998 is mathematically valid for Mirna's 3-class classification problem...
but it's **evaluating the wrong problem entirely**.

---

## Summary Table: Decision Consequences

| Decision Point | Mirna's Choice | Consequence | Correct Choice | Consequence |
|---|---|---|---|---|
| Problem Type | Classification | Cannot output continuous risk | Regression | Outputs continuous risk ✓ |
| Risk Threshold | -20, -10 arbitrary | Creates 90%+ imbalance | -6 per ESA | Creates 90% imbalance ✓ |
| Model | Classifier | Outputs discrete labels | Regressor | Outputs continuous values ✓ |
| F2 Scope | 3-class weighted | F2 = 0.9998 (fake) | Binary (-6 threshold) | F2 = 0.25 (real) ✓ |
| MSE Scope | All test data | Dominated by easy cases | High-risk only | Penalizes collisions ✓ |
| Loss Formula | MSE / F2 | L = 0.0134 (invalid) | (1/F2) × MSE | L = 58.4 (valid) ✓ |

---

## The Path Forward

For Mirna to achieve valid, competitive results:

```
CURRENT STATE (WRONG):
  Classification model predicting 3 classes
         ↓
         └→ Cannot be evaluated on ESA continuous metric
            Score is mathematically invalid for challenge

TRANSITION (WEEK 1):
  Switch to regression model with continuous predictions
         ↓
         └→ Can be evaluated using ESA binary threshold
            Scores become comparable to baseline

TARGET STATE (WEEK 2-3):
  Add advanced feature engineering and optimization
         ↓
         └→ Achieve competitive ESA Loss (< 58)
            Ready for official challenge submission
```

---

## Conclusion

Mirna's F2 score of **0.9998 is not "better" than the baseline 0.25**—they're measuring **different problems**.

It's like comparing:
- 🏃 Mirna: "I can run 100m in 2 seconds on a 50m track"
- 🏃 Baseline: "I can run 100m in 9.58 seconds"

Mirna's result is physically implausible because the distances don't match.
Similarly, Mirna's score is metrically implausible for the ESA challenge.

**The solution is clear: Reframe the entire problem as continuous regression.**

