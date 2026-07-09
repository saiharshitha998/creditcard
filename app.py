import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below.")

# Manual Input
time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

inputs = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    inputs.append(value)

if st.button("Predict"):

    data = [[
        time,
        *inputs,
        amount
    ]]

    columns = [
        "Time",
        "V1","V2","V3","V4","V5","V6","V7","V8","V9",
        "V10","V11","V12","V13","V14","V15","V16",
        "V17","V18","V19","V20","V21","V22","V23",
        "V24","V25","V26","V27","V28",
        "Amount"
    ]

    df = pd.DataFrame(data, columns=columns)

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]

    confidence = max(probability) * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"**Confidence:** {confidence:.2f}%")
