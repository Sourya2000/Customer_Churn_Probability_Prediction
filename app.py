import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# 1. Load the pre-trained artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_prediction_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

# Define the exact feature order used in training
FEATURE_COLUMNS = [
    'State', 'Account Length', 'Area Code', "Int'l Plan", 'VMail Plan',
    'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge', 'Eve Mins',
    'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge',
    'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls',
    'Total_Charge', 'Avg_Charge_Per_Min', 'High_Service_Calls'
]

st.title("🎯 Customer Churn Prediction")
st.markdown("Predict the likelihood of a customer leaving your service based on their usage patterns.")

# 2. Sidebar for User Inputs
st.sidebar.header("Customer Information")

def get_user_input():
    # Categorical Inputs
    state = st.sidebar.selectbox("State", label_encoders['State'].classes_)
    intl_plan = st.sidebar.selectbox("International Plan", ["no", "yes"])
    vmail_plan = st.sidebar.selectbox("VMail Plan", ["no", "yes"])
    
    # Numeric Inputs
    acc_len = st.sidebar.number_input("Account Length", min_value=1, value=100)
    area_code = st.sidebar.selectbox("Area Code", [408, 415, 510])
    vmail_msg = st.sidebar.number_input("VMail Messages", min_value=0, value=0)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        day_mins = st.number_input("Day Mins", value=180.0)
        day_calls = st.number_input("Day Calls", value=100)
        eve_mins = st.number_input("Eve Mins", value=200.0)
        eve_calls = st.number_input("Eve Calls", value=100)
    with col2:
        night_mins = st.number_input("Night Mins", value=200.0)
        night_calls = st.number_input("Night Calls", value=100)
        intl_mins = st.number_input("Intl Mins", value=10.0)
        intl_calls = st.number_input("Intl Calls", value=3)
        
    cust_serv = st.sidebar.slider("Customer Service Calls", 0, 10, 1)

    # Manual calculation of engineered features from the notebook
    day_charge = day_mins * 0.1700  # Approximated rates from typical dataset
    eve_charge = eve_mins * 0.0850
    night_charge = night_mins * 0.0450
    intl_charge = intl_mins * 0.2700
    
    total_charge = day_charge + eve_charge + night_charge + intl_charge
    total_mins = day_mins + eve_mins + night_mins + intl_mins
    avg_charge_per_min = total_charge / total_mins if total_mins > 0 else 0
    high_service_calls = 1 if cust_serv >= 4 else 0

    data = {
        'State': state, 'Account Length': acc_len, 'Area Code': area_code,
        "Int'l Plan": intl_plan, 'VMail Plan': vmail_plan, 'VMail Message': vmail_msg,
        'Day Mins': day_mins, 'Day Calls': day_calls, 'Day Charge': day_charge,
        'Eve Mins': eve_mins, 'Eve Calls': eve_calls, 'Eve Charge': eve_charge,
        'Night Mins': night_mins, 'Night Calls': night_calls, 'Night Charge': night_charge,
        'Intl Mins': intl_mins, 'Intl Calls': intl_calls, 'Intl Charge': intl_charge,
        'CustServ Calls': cust_serv, 'Total_Charge': total_charge,
        'Avg_Charge_Per_Min': avg_charge_per_min, 'High_Service_Calls': high_service_calls
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# 3. Prediction Logic
if st.button("Analyze Customer"):
    # Preprocessing
    processed_df = input_df.copy()
    for col in ['State', "Int'l Plan", 'VMail Plan']:
        processed_df[col] = label_encoders[col].transform(processed_df[col])
    
    # Scale and Predict
    scaled_data = scaler.transform(processed_df[FEATURE_COLUMNS])
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    # 4. Display Results
    st.subheader("Prediction Results")
    col_res1, col_res2 = st.columns(2)
    
    if prediction == 1:
        col_res1.error("🚨 HIGH RISK OF CHURN")
    else:
        col_res1.success("✅ LOW RISK OF CHURN")
        
    col_res2.metric("Churn Probability", f"{probability:.2%}")
    
    # Insights based on features
    if input_df['CustServ Calls'].iloc[0] >= 4:
        st.warning("High number of customer service calls detected. This is a primary driver for churn.")