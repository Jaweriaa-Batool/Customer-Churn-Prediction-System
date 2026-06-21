import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("🎯 Telecom Customer Churn Prediction System")
st.write("This interactive system analyzes customer demographics and account info to predict the probability of a customer leaving the service.")

csv_filename = "telco_churn.csv"

if not os.path.exists(csv_filename):
    st.error(f"❌ Error: `{csv_filename}` file not found.")
    st.stop()

@st.cache_data
def load_clean_churn_data():
    df = pd.read_csv(csv_filename)
    
    df = df.drop(columns=['customerID'], errors='ignore')
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
   
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        
    return df

df = load_clean_churn_data()

if st.checkbox("Show Telco Dataset Cleaned Preview (First 10 Rows)"):
    st.dataframe(df.head(10))

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@st.cache_resource
def train_churn_models(X_tr, y_tr, X_te, y_te):
 
    lr = LogisticRegression(random_state=42, class_weight='balanced').fit(X_tr, y_tr)
    lr_preds = lr.predict(X_te)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced').fit(X_tr, y_tr)
    rf_preds = rf.predict(X_te)
    
    metrics = {
        'Model Algorithm': ['Logistic Regression', 'Random Forest Classifier'],
        'Accuracy': [accuracy_score(y_te, lr_preds), accuracy_score(y_te, rf_preds)],
        'Precision': [precision_score(y_te, lr_preds), precision_score(y_te, rf_preds)],
        'Recall (Catch Rate)': [recall_score(y_te, lr_preds), recall_score(y_te, rf_preds)]
    }
    return pd.DataFrame(metrics), lr, rf

metrics_df, lr_model, rf_model = train_churn_models(X_train_scaled, y_train, X_test_scaled, y_test)

st.sidebar.header("Model & Prediction Settings")
selected_model = st.sidebar.selectbox("Choose Core Prediction Model", ["Random Forest Classifier", "Logistic Regression"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Performance Matrix Comparison")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))

with col2:
    st.subheader("📈 Model Accuracy vs Recall Overview")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    melted_metrics = pd.melt(metrics_df, id_vars=['Model Algorithm'], value_vars=['Accuracy', 'Recall (Catch Rate)'])
    sns.barplot(x='Model Algorithm', y='value', hue='variable', data=melted_metrics, ax=ax, palette='Set2')
    ax.set_ylim(0, 1.0)
    st.pyplot(fig)

st.markdown("---")

st.subheader("🔮 Live Risk Assessment Simulation (Interactive Controls)")
st.write("Adjust the client metrics below to evaluate risk levels instantly:")

col_u1, col_u2, col_u3 = st.columns(3)

with col_u1:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tech_support = st.selectbox("Tech Support Subscribed", ["No", "Yes", "No internet service"])
    tenure = st.slider("Tenure (Months with Company)", min_value=1, max_value=72, value=12)

with col_u2:
    internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Mode", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18, max_value=120, value=65)

with col_u3:
    online_security = st.selectbox("Online Security Feature", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    total_charges = st.number_input("Total Historical Charges ($)", min_value=18.0, max_value=8500.0, value=780.0)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
tech_map = {"No": 0, "No internet service": 1, "Yes": 2}
security_map = {"No": 0, "No internet service": 1, "Yes": 2}
payment_map = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}
paperless_map = {"No": 0, "Yes": 1}

input_data = [
    1, # gender (dummy)
    0, # SeniorCitizen (dummy)
    1, # Partner (dummy)
    0, # Dependents (dummy)
    tenure,
    1, # PhoneService (dummy)
    0, # MultipleLines (dummy)
    internet_map[internet_service],
    security_map[online_security],
    0, # OnlineBackup (dummy)
    0, # DeviceProtection (dummy)
    tech_map[tech_support],
    0, # StreamingTV (dummy)
    0, # StreamingMovies (dummy)
    contract_map[contract],
    paperless_map[paperless],
    payment_map[payment_method],
    monthly_charges,
    total_charges
]

# Scaler reshaping boundary
input_scaled = scaler.transform([input_data])

if selected_model == "Logistic Regression":
    prob = lr_model.predict_proba(input_scaled)[0][1]
else:
    prob = rf_model.predict_proba(input_scaled)[0][1]

st.write("### 🚨 Churn Assessment Report:")
if prob > 0.50:
    st.error(f"🟥 **High Risk Customer Alert!** — Churn Probability: **{prob:.2%}**")
    st.warning("💡 **Business Recommendation:** This customer is highly likely to leave. Offer a loyalty discount or long-term contract upgrade immediately.")
else:
    st.success(f"🟩 **Low Risk/Loyal Customer** — Churn Probability: **{prob:.2%}**")
    st.info("💡 **Business Recommendation:** Safe retention. Keep engaging with standard marketing newsletters.")
