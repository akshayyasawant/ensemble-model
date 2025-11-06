import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor — Ensemble ML")

MODEL_PATH = "models/ensemble_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load model
try:
    model = load_model()
    st.success(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.caption("This app uses your saved preprocessing + ensemble model (pipeline) to estimate car `selling_price`.")

# Try to infer training feature names
def get_feature_names_from_model(m):
    try:
        return list(m.feature_names_in_)
    except Exception:
        return None

feature_cols = get_feature_names_from_model(model)

# Fallback for typical CarDekho columns if still unknown
if feature_cols is None:
    feature_cols = [
        "name","year","km_driven","fuel","seller_type","transmission","owner","brand",
        "mileage","engine","max_power","seats"
    ]

# --------------- Single Prediction Section ----------------
st.header("🔮 Predict Selling Price")
st.write("Enter car details below to get an estimated selling price.")

values = {}
fuel_choices   = ["Petrol","Diesel","CNG","LPG","Electric"]
seller_choices = ["Individual","Dealer","Trustmark Dealer"]
trans_choices  = ["Manual","Automatic"]
owner_choices  = ["First Owner","Second Owner","Third Owner","Fourth & Above Owner","Test Drive Car"]
brand_choices  = ["Maruti","Hyundai","Honda","Toyota","Tata","Mahindra","Ford","Volkswagen","Renault","Skoda"]

for col in feature_cols:
    low = col.lower()
    if low == "year":
        values[col] = st.number_input("Year", 1990, 2025, 2016, step=1)
    elif low == "km_driven":
        values[col] = st.number_input("Kilometers driven", 0, 10_000_000, 42000, step=500)
    elif low == "fuel":
        values[col] = st.selectbox("Fuel", fuel_choices)
    elif low == "seller_type":
        values[col] = st.selectbox("Seller Type", seller_choices)
    elif low == "transmission":
        values[col] = st.selectbox("Transmission", trans_choices)
    elif low == "owner":
        values[col] = st.selectbox("Owner", owner_choices)
    elif low == "brand":
        values[col] = st.selectbox("Brand", brand_choices)
    elif low == "name":
        values[col] = st.text_input("Model Name", "Maruti Swift")
    elif low in ["mileage","engine","max_power","seats"]:
        default = 20.0 if low=="mileage" else (1197.0 if low=="engine" else (82.0 if low=="max_power" else 5))
        values[col] = st.number_input(col.capitalize(), value=float(default))
    else:
        values[col] = st.text_input(col.capitalize())

if st.button("💰 Predict Price", type="primary"):
    X_new = pd.DataFrame([values])
    try:
        pred = model.predict(X_new)[0]
        st.success(f"💡 Estimated selling price: **₹{pred:,.0f}**")
    except Exception as e:
        st.error(f"Prediction failed. Check that columns match your training schema.\n{e}")

st.markdown("---")
st.caption("Developed for Experiment 6 — Ensemble Machine Learning (Regression)")
