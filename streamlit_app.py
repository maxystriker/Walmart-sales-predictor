
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon=":shopping_cart:",
    layout="wide"
)

# Load model with encoding info
@st.cache_resource
def load_model():
    model_data = joblib.load('optimized_model.joblib')
    return model_data

model_data = load_model()
model = model_data['model']
type_mapping = model_data['type_mapping']

# App header
st.title("Walmart Weekly Sales Predictor")
st.markdown("Predict sales for any store department with 94.5% accuracy!")

# Input form
col1, col2 = st.columns(2)

with col1:
    st.header("Store Information")
    store = st.number_input("Store Number", min_value=1, max_value=100, value=1)
    dept = st.number_input("Department", min_value=1, max_value=100, value=1)
    store_type = st.selectbox("Store Type", ["A", "B", "C"], help="A=Large, B=Medium, C=Small")
    size = st.number_input("Store Size (sq ft)", min_value=10000, max_value=500000, value=150000)

with col2:
    st.header("Economic Factors")
    temperature = st.slider("Temperature (°F)", -20.0, 120.0, 42.3)
    fuel_price = st.slider("Fuel Price ($)", 1.0, 5.0, 2.5)
    cpi = st.slider("CPI", 100.0, 250.0, 210.0)
    unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 8.0)

# Predict button
if st.button("Predict Weekly Sales", type="primary"):
    try:
        # Convert store type to number
        type_encoded = type_mapping[store_type]
        
        # Prepare input data in correct order
        input_data = pd.DataFrame([{
            'Store': store,
            'Dept': dept,
            'Type': type_encoded,
            'Temperature': temperature,
            'Fuel_Price': fuel_price,
            'Size': size,
            'CPI': cpi,
            'Unemployment': unemployment,
            'MarkDown1': 0,
            'MarkDown2': 0,
            'MarkDown3': 0,
            'MarkDown4': 0,
            'MarkDown5': 0
        }])
        
        # Ensure correct column order
        input_data = input_data[model_data['feature_names']]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        st.success("Prediction Complete!")
        st.metric(label="Predicted Weekly Sales", value=f"${prediction:,.2f}")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please check all inputs are valid")

# Footer
st.markdown("---")
st.caption("Built with Machine Learning | 94.5% Accuracy")
