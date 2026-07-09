import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details and click **Predict**.")

# Manual Inputs
amount = st.number_input("Amount", value=0.0)
v14 = st.number_input("V14", value=0.0)
v12 = st.number_input("V12", value=0.0)
v10 = st.number_input("V10", value=0.0)
v17 = st.number_input("V17", value=0.0)
v4 = st.number_input("V4", value=0.0)

if st.button("Predict"):

    data = pd.DataFrame({
        "V14": [v14],
        "V12": [v12],
        "V10": [v10],
        "V17": [v17],
        "Amount": [amount],
        "V4": [v4]
    })

    prediction = model.predict(data)[0]

    confidence = model.predict_proba(data).max() * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.write(f"**Confidence:** {confidence:.2f}%")
