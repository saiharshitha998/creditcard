import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection System")
st.write("Upload a CSV file containing transaction data.")

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:
        predictions = model.predict(df)

        df["Prediction"] = predictions

        df["Prediction"] = df["Prediction"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        st.subheader("Prediction Results")
        st.dataframe(df.head())

        fraud_count = (df["Prediction"] == "Fraud").sum()
        legit_count = (df["Prediction"] == "Legitimate").sum()

        st.metric("Fraud Transactions", fraud_count)
        st.metric("Legitimate Transactions", legit_count)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "prediction_results.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")