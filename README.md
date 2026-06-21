# 🎯 Telecom Customer Churn Prediction System

An end-to-end Machine Learning and Business Intelligence system built to predict customer churn risk in the telecom sector. This project features an interactive live risk simulation dashboard built with **Streamlit** and hosted via Google Colab.

## 🚀 Features
- **Data Inspection:** Cleaned and preprocessed view of the real-world Kaggle/IBM Telco Customer Churn dataset.
- **Dual-Model Cross Comparison:**
  - **Logistic Regression:** Optimized with balanced class weights, focusing on higher Recall to catch maximum potential churn risks.
  - **Random Forest Classifier:** An advanced ensemble model utilizing non-linear decision trees to capture complex customer behavior patterns for high overall accuracy.
- **Live Risk Assessment Simulation:** An interactive click-and-slide control panel allowing users to adjust client parameters (tenure, contract, payment mode) and observe real-time probability shifts and automated business recommendations.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Scikit-Learn (StandardScaler, LabelEncoder), Matplotlib, Seaborn
- **Framework:** Streamlit Dashboard Engine
- **Environment:** Google Colab Cloud Ecosystem

## 📂 Repository Structure
├── Customer_Churn_Predictor.ipynb   # Main Google Colab Notebook
├── app.py                           # Streamlit Dashboard UI Engine
└── README.md                        # Project Documentation
