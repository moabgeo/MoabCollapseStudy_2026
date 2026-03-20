import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# 1. Load and Prepare Data
df = pd.read_csv('collapse_data.csv')

# Use nomenclature from the paper
features = [
    'Initial_Void_Ratio_e0', 
    'Initial_Dry_Unit_Weight_pcf', 
    'Initial_Saturation_S0', 
    'Initial_Moisture_w0', 
    'Applied_Surcharge_psf'
]

X = df[features]

# Define target based on the 1% threshold (as described in Model Performance)
threshold_value = 1.0
y_binary = (df['Percent_Collapse'] > threshold_value).astype(int)

# 2. Sequential Data Splitting (70/15/15)
# First split: 70% Train, 30% Temporary (Validation + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.30, random_state=42, stratify=y_binary
)

# Second split: Split the 30% into two equal halves (15% and 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 3. Initialize XGBoost Classifier with Paper Parameters
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,           # "Sweet spot" for complexity
    learning_rate=0.3,     # Controls shrinkage/stability
    objective='binary:logistic', # Required for Log-Loss
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 4. Performance Metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")

# 5. Calculate SHAP Values for Figure 5
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Calculate relative significance (normalized to 1.0 for comparison)
raw_importance = np.abs(shap_values.values).mean(0)
# Normalizing to match the "Relative Coefficient" described in the paper
scaled_importance = (raw_importance / raw_importance.sum()) 

# 6. Figure 5: Comparison of Significance
traditional_r2 = [0.32, 0.29, 0.26, 0.18, 0.10] # Values from Paper

fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(features))
width = 0.35

ax.barh(ind, traditional_r2, width, label='Traditional Linear R²', color='lightgrey')
ax.barh(ind + width, scaled_importance, width, label='ML Relative Significance', color='#4c72b0')

ax.set_xlabel('Relative Significance Coefficient')
ax.set_title('Figure 5: Traditional Statistical R² vs. ML Feature Significance')
ax.set_yticks(ind + width / 2)
ax.set_yticklabels(features)
ax.legend()
plt.tight_layout()
plt.savefig('figure_5_comparison.png')

# 7. Figure 6: Sensitivity Analysis across Thresholds (1%, 2%, 5%)
thresholds = [1.0, 2.0, 5.0]
sensitivity_results = []

for t in thresholds:
    y_t = (df['Percent_Collapse'] > t).astype(int)
    m_t = xgb.XGBClassifier(max_depth=6, learning_rate=0.3, objective='binary:logistic', random_state=42)
    m_t.fit(X, y_t)
    
    # Use SHAP to get importance for this specific threshold
    exp_t = shap.Explainer(m_t)
    sv_t = exp_t(X)
    importance_t = np.abs(sv_t.values).mean(0)
    sensitivity_results.append(importance_t / importance_t.sum())

# Plot Sensitivity
fig6, ax6 = plt.subplots(figsize=(10, 6))
for i, feat in enumerate(features):
    ax6.plot(thresholds, [res[i] for res in sensitivity_results], marker='o', label=feat)

ax6.set_xlabel('Collapse Threshold (%)')
ax6.set_ylabel('Relative Feature Importance')
ax6.set_title('Figure 6: Sensitivity of Feature Importance across Thresholds')
ax6.set_xticks(thresholds)
ax6.legend()
plt.tight_layout()
plt.savefig('figure_6_sensitivity.png')
