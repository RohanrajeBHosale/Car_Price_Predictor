import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the Trained Model and Column Names
# We use @st.cache_resource so the model loads once and stays in memory (faster)
@st.cache_resource
def load_model():
    model = joblib.load("data/car_price_model.pkl")
    model_columns = joblib.load("data/model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_model()
except FileNotFoundError:
    st.error("Model files not found! Please run 'step3_train_model.py' first.")
    st.stop()

# 2. App Title and Description
st.set_page_config(page_title="Used Car Price AI", layout="centered")
st.title("🚗 AI Used Car Price Predictor")
st.markdown("Enter the car details below to get an estimated market value using XGBoost.")

# 3. User Input Form
with st.form("prediction_form"):
    st.subheader("Vehicle Details")
    col1, col2 = st.columns(2)

    with col1:
        # We list the most common manufacturers to keep the UI clean
        manufacturer = st.selectbox("Brand", [
            'ford', 'chevrolet', 'toyota', 'honda', 'nissan', 'jeep',
            'bmw', 'mercedes-benz', 'audi', 'hyundai', 'kia', 'subaru',
            'volkswagen', 'dodge', 'ram', 'gmc'
        ])
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)
        odometer = st.number_input("Mileage (Odometer)", min_value=0, max_value=300000, value=50000)
        cylinders = st.selectbox("Cylinders", ['4 cylinders', '6 cylinders', '8 cylinders', 'other'])

    with col2:
        fuel = st.selectbox("Fuel Type", ['gas', 'diesel', 'hybrid', 'electric'])
        transmission = st.selectbox("Transmission", ['automatic', 'manual', 'other'])
        drive = st.selectbox("Drive Type", ['4wd', 'fwd', 'rwd'])
        type_ = st.selectbox("Car Type", ['sedan', 'SUV', 'truck', 'pickup', 'coupe', 'hatchback', 'convertible', 'van'])

    # Submit Button
    submitted = st.form_submit_button("💰 Predict Price", type="primary")

# 4. Prediction Logic
if submitted:
    # A. Feature Engineering (match what we did in Step 2)
    car_age = 2026 - year

    # B. Create a template input with 0s for all columns the model expects
    input_data = {col: 0 for col in model_columns}

    # C. Fill in the numeric values
    input_data['odometer'] = odometer
    input_data['car_age'] = car_age

    # D. One-Hot Encoding Logic
    # The model has columns like 'manufacturer_ford'. We need to set that specific column to 1.
    def set_category(cat_name, value):
        col_name = f"{cat_name}_{value}"
        if col_name in input_data:
            input_data[col_name] = 1

    set_category('manufacturer', manufacturer)
    set_category('fuel', fuel)
    set_category('transmission', transmission)
    set_category('drive', drive)
    set_category('type', type_)
    set_category('cylinders', cylinders)

    # E. Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # F. Predict
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Market Value: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
