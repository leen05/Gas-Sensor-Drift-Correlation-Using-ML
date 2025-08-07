
# ⛽️ Gas Sensor Drift Prediction using XGBoost

This project focuses on predicting gas concentrations from sensor data while addressing the issue of **sensor drift** — a common problem in long-term gas sensing systems. The model is trained on preprocessed batch-wise data and uses **XGBoost**, a powerful and efficient gradient boosting algorithm.

---

## 📌 Description

Gas sensors tend to degrade over time due to environmental changes like temperature and humidity, causing inaccurate readings. This project tackles that problem by:

- Preprocessing raw sensor data from multiple batches
- Building an XGBoost regression model for gas concentration prediction
- Evaluating performance using metrics like MAE and RMSE
- Visualizing trends using heatmaps, histograms, scatter plots, and boxplots

This solution is useful in fields like environmental monitoring, industrial safety, and IoT-based gas detection systems.

---

## 🧠 Features

- Machine learning with **XGBoost**
- Batch-wise analysis to study sensor drift
- Clean preprocessing pipeline
- Evaluation metrics: MAE, RMSE, R²
- Data visualizations to study feature relationships and performance

---

## Visualisation Through Streamlit  

The Streamlit dashboard enables interactive exploration of sensor drift and model predictions. It features:

- 📁 Dataset folder input to load batch-wise gas sensor data
- 📊 Pie chart of gas type distribution across batches
- 📉 Histograms and box plots of sensor features
- 🌡️ Heatmaps to visualize feature correlations
- 📈 Scatter and residual plots for predicted vs actual gas concentrations
- 🧮 Real-time evaluation using R², MAE, and RMSE
- ⚙️ Fully responsive interface for monitoring and analysis

--- 

## 🛠️ Tech Stack

- Python 3.10+
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly, Altair
- scikit-learn

---



