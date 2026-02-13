import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load your regional dataset
df = pd.read_csv('collapse_data.csv')
X = df[['Void_Ratio_e0', 'Dry_Unit_Weight_pcf', 'Saturation_S0', 'Moisture_Content_w0', 'Applied_Load_psf']]
y = df['Percent_Collapse']

# 2. Cross-Validation (Better for n=86 than a single split)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    scores.append(r2_score(y_test, preds))

print(f"Average XAI R2 Score: {np.mean(scores):.3f}")

# 3. Final Model for Interpretation
final_model = xgb.XGBRegressor().fit(X, y)

# 4. SHAP Interpretation (The "Forensic Thought Process")
explainer = shap.Explainer(final_model)
shap_values = explainer(X)

# Generate Summary Plot
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Feature Importance: Drivers of Soil Collapse")
plt.show()

# Generate Dependence Plot for Void Ratio (to see the "mitigated" relationship)
shap.dependence_plot("Void_Ratio_e0", shap_values.values, X)
