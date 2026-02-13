# Regional Soil Collapse Prediction using Explainable AI (XAI)

This repository contains the regional dataset and Python implementation for analyzing soil collapse potential in the Colorado Plateau (Moab, UT region). 

## Project Overview
Standard geotechnical practice often relies on limited oedometer testing (ASTM D4546). This project utilizes a regional dataset of 86 collapse tests to identify critical drivers of soil collapse using:
* **XGBoost:** For high-performance predictive modeling.
* **SHAP (SHapley Additive exPlanations):** To provide a forensic "thought process" and interpret how physical properties influence the model's results.

## Dataset
The dataset consists of 86 regional samples featuring:
* Void Ratio ($e_0$)
* Dry Unit Weight ($\gamma_{d0}$)
* Saturation ($S_0$)
* Moisture Content ($w_0$)
* Applied Load

## Requirements
To run the analysis, you will need:
```bash
pip install pandas xgboost shap scikit-learn matplotlib
