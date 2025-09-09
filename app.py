import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Insurance Claims Risk Analyzer", layout="wide")

st.title("📊 Insurance Claims Risk Analyzer")
st.markdown("Analyze insurance claim data, calculate KPIs, and predict fraud risk.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Insurance Claims CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # --- Data Preparation ---
    df = df.copy()
    df["LossRatio"] = df["ClaimAmount"] / df["Premium"].replace(0, np.nan)
    df["LossRatio"] = df["LossRatio"].fillna(0)

    # --- KPIs ---
    total_claims = len(df)
    avg_premium = round(df["Premium"].mean(), 2)
    avg_claim = round(df["ClaimAmount"].mean(), 2)
    overall_loss_ratio = round(df["LossRatio"].mean(), 2)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Claims", total_claims)
    kpi2.metric("Avg Premium", f"₹{avg_premium}")
    kpi3.metric("Avg Claim Amt", f"₹{avg_claim}")
    kpi4.metric("Loss Ratio", overall_loss_ratio)

    # --- Charts ---
    st.subheader("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Claim Frequency by Region**")
        fig, ax = plt.subplots()
        sns.countplot(x="Region", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Fraud vs Non-Fraud**")
        fraud_counts = df["IsFraud"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(fraud_counts, labels=["Non-Fraud", "Fraud"], autopct="%1.1f%%")
        ax.set_title("Fraud Distribution")
        st.pyplot(fig)

    st.markdown("**Loss Ratio Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df["LossRatio"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # --- Fraud Prediction Model ---
    st.subheader("🤖 Fraud Prediction Model")

    features = ["CustomerAge", "VehicleAge", "PolicyTenure", "Premium"]
    if all(col in df.columns for col in features):
        X = df[features]
        y = df["IsFraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f"**Model Accuracy:** {acc:.2f}")

        if st.button("Predict Fraud on All Data"):
            df["FraudPrediction"] = model.predict(X)
            st.dataframe(df[["ClaimID", "CustomerAge", "Premium", "ClaimAmount", "FraudPrediction"]].head())
            st.success("Fraud predictions added!")

    else:
        st.warning("Dataset missing required features for fraud prediction.")

else:
    st.info("👆 Upload a CSV file to start analysis.")
