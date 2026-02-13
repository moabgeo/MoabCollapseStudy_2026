import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 1. Load Data
df = pd.read_csv('collapse_data.csv')
features = ['Void_Ratio_e0', 'Dry_Unit_Weight_pcf', 'Saturation_S0', 'Moisture_Content_w0', 'Applied_Load_psf']
X = df[features]
y = df['Percent_Collapse']

# 2. Train Model and Calculate Global R2
model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X, y)
total_r2 = r2_score(y, model.predict(X)) # This is the "Total Significance"

# 3. Calculate SHAP Values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 4. Scale SHAP Importance to the Total R2
# Mean absolute SHAP values give us the raw importance
raw_importance = np.abs(shap_values.values).mean(0)
# Scale them so the sum of the bars equals the total R2 of the model
scaled_importance = (raw_importance / raw_importance.sum()) * total_r2

# 5. Create Figure 5: Comparison of Significance
traditional_r2 = [0.316, 0.294, 0.262, 0.183, 0.100] # Values from your analysis

fig, ax = plt.subplots(figsize=(10, 6))
ind = np.arange(len(features))
width = 0.35

ax.barh(ind, traditional_r2, width, label='Traditional Linear R²', color='lightgrey')
ax.barh(ind + width, scaled_importance, width, label='ML Contribution to R²', color='#4c72b0')

ax.set_xlabel('Significance Value (R²)')
ax.set_title('Figure 5: Traditional Statistical R² vs. ML Feature Significance')
ax.set_yticks(ind + width / 2)
ax.set_yticklabels(features)
ax.legend()

plt.tight_layout()
plt.show()
