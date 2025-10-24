import streamlit as st
import joblib
import numpy as np
import pandas as pd
import mlflow
import os

# Use a local folder for MLflow tracking
mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns').replace('\\', '/')}")
mlflow.set_experiment("Motorbike_Price_Prediction_App")

# --- Load model ---
MODEL_PATH= "../models/final_rf_pipeline.joblib"

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Motorbike Price Predictor", layout="centered")

st.title("Motorbike Price Prediction in Sri Lanka")

# --- Collect input from user ---
col1, col2 = st.columns(2)
with col1:
    brand = st.text_input("Brand", "Honda")
    model_name = st.text_input("Model", "CB125")
    bike_type = st.selectbox("Bike Type", ["Motorcycle", "Scooter", "Sports", "Cruiser", "Other"])
    seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
with col2:
    capacity = st.number_input("Capacity (CC)", 50, 2000, 125)
    mileage = st.number_input("Mileage (km)")
    age = st.number_input("Make (year)", 0, 2025, 5)


# --- Create a DataFrame from user input ---
input_data = pd.DataFrame([{
    'Brand': brand,
    'Model': model_name,
    'Bike Type': bike_type,
    'Capacity': capacity,
    'Mileage': mileage,
    'Year': age,
    'Seller': seller,
}])

st.markdown("### Input Summary")
st.dataframe(input_data)

# --- Predict ---
if st.button("🔮 Predict Price"):
    try:
        pred = model.predict(input_data)[0]

        st.success(f"💰 Estimated Price: LKR **{pred:,.0f}**")

        # Log prediction to MLflow
        with mlflow.start_run(run_name="single_prediction", nested=True):
            mlflow.log_params(input_data.iloc[0].to_dict())  # user input
            mlflow.log_param("model_name", os.path.basename(MODEL_PATH))
            mlflow.log_artifact(MODEL_PATH, artifact_path="model")
            mlflow.log_metric("predicted_price", float(pred))
            mlflow.set_tag("source", "streamlit_app")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

