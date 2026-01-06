import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- 1. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    # Ensure these files are in your project folder
    model = joblib.load('churn_prediction_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

# Exact order from your feature_scaler.pkl
FEATURE_COLUMNS = [
    'State', 'Account Length', 'Area Code', "Int'l Plan", 'VMail Plan',
    'VMail Message', 'Day Mins', 'Day Calls', 'Day Charge', 'Eve Mins',
    'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge',
    'Intl Mins', 'Intl Calls', 'Intl Charge', 'CustServ Calls',
    'Total_Charge', 'Avg_Charge_Per_Min', 'High_Service_Calls'
]

# --- 2. PDF GENERATION FUNCTION ---
def create_pdf(prediction, probability, dynamic_actions, customer_id="New Customer"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Customer Churn Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    status = "HIGH CHURN RISK" if prediction == 1 else "STABLE / LOW RISK"
    pdf.cell(200, 10, txt=f"Analysis for: {customer_id}", ln=True)
    pdf.cell(200, 10, txt=f"Status: {status}", ln=True)
    pdf.cell(200, 10, txt=f"Churn Probability: {probability:.2%}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Top Risk Drivers & Recommended Actions:", ln=True)
    
    pdf.set_font("Arial", size=11)
    for feature, action in dynamic_actions:
        pdf.ln(5)
        pdf.multi_cell(0, 10, txt=f"Feature: {feature.upper()}\nAction: {action}")
        
    return pdf.output(dest='S').encode('latin-1')

# --- 3. UI LAYOUT & INPUTS ---
st.title("🎯 Customer Churn Intelligence Dashboard")

st.sidebar.header("Customer Input Data")

def get_user_input():
    # Categorical Inputs
    state = st.sidebar.selectbox("State", label_encoders['State'].classes_)
    intl_plan = st.sidebar.selectbox("International Plan", ["no", "yes"])
    vmail_plan = st.sidebar.selectbox("VMail Plan", ["no", "yes"])
    
    # Basic Info
    acc_len = st.sidebar.number_input("Account Length (Months)", min_value=1, value=100)
    area_code = st.sidebar.selectbox("Area Code", [408, 415, 510])
    vmail_msg = st.sidebar.number_input("VMail Messages", min_value=0, value=0)
    
    st.subheader("Usage Patterns")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Daytime**")
        day_mins = st.number_input("Day Mins", value=180.0)
        day_calls = st.number_input("Day Calls", value=100)
    with col2:
        st.markdown("**Evening**")
        eve_mins = st.number_input("Eve Mins", value=200.0)
        eve_calls = st.number_input("Eve Calls", value=100) # Added here
    with col3:
        st.markdown("**Night/Intl**")
        night_mins = st.number_input("Night Mins", value=200.0)
        intl_mins = st.number_input("Intl Mins", value=10.0)

    cust_serv = st.sidebar.slider("Customer Service Calls", 0, 10, 1)

    # Logic-based Feature Engineering (Calculated fields)
    day_charge = day_mins * 0.17
    eve_charge = eve_mins * 0.085
    night_charge = night_mins * 0.045
    intl_charge = intl_mins * 0.27
    
    total_charge = day_charge + eve_charge + night_charge + intl_charge
    total_mins = day_mins + eve_mins + night_mins + intl_mins
    avg_charge_per_min = total_charge / total_mins if total_mins > 0 else 0
    high_service_calls = 1 if cust_serv >= 4 else 0

    data = {
        'State': state, 'Account Length': acc_len, 'Area Code': area_code,
        "Int'l Plan": intl_plan, 'VMail Plan': vmail_plan, 'VMail Message': vmail_msg,
        'Day Mins': day_mins, 'Day Calls': day_calls, 'Day Charge': day_charge,
        'Eve Mins': eve_mins, 'Eve Calls': eve_calls, 'Eve Charge': eve_charge,
        'Night Mins': night_mins, 'Night Calls': 100, 'Night Charge': night_charge,
        'Intl Mins': intl_mins, 'Intl Calls': 3, 'Intl Charge': intl_charge,
        'CustServ Calls': cust_serv, 'Total_Charge': total_charge,
        'Avg_Charge_Per_Min': avg_charge_per_min, 'High_Service_Calls': high_service_calls
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# --- 4. PREDICTION & DYNAMIC RECOMMENDATIONS ---
if st.button("Generate Detailed Analysis"):
    # Preprocessing
    processed_df = input_df.copy()
    for col in ['State', "Int'l Plan", 'VMail Plan']:
        processed_df[col] = label_encoders[col].transform(processed_df[col])
    
    # Scaling and Inference
    scaled_data = scaler.transform(processed_df[FEATURE_COLUMNS])
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    # Result Visualization
    st.divider()
    st.subheader("Analysis Summary")
    c1, c2 = st.columns(2)
    if prediction == 1:
        c1.error("🚨 HIGH CHURN RISK DETECTED")
    else:
        c1.success("✅ STABLE CUSTOMER PROFILE")
    c2.metric("Risk Probability", f"{probability:.2%}")

    # Calculate Local Impact (Dynamic Logic)
    # We find which specific inputs are driving the prediction for THIS user
    local_impact = np.abs(scaled_data[0] * model.feature_importances_)
    impact_df = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'impact': local_impact
    }).sort_values('impact', ascending=False)

    st.header("🚀 Personalized Business Action Plan")
    
    dynamic_actions_list = []
    top_3_drivers = impact_df.head(3)
    
    for _, row in top_3_drivers.iterrows():
        feat = row['feature']
        raw_val = input_df[feat].iloc[0]
        
        # Branching logic for dynamic recommendations
        if "Charge" in feat or "Mins" in feat:
            action = f"Usage volume in {feat} ({raw_val}) is excessive. Recommend transition to an Unlimited Tier plan."
        elif "Calls" in feat:
            action = f"Frequent interactions ({raw_val} {feat}) suggest friction. Schedule a 1-on-1 account health review."
        elif "Plan" in feat:
            action = f"Customer's {feat} status is poorly optimized for their usage. Propose a plan realignment."
        else:
            action = "Monitor usage weekly and apply a proactive 10% loyalty credit."
        
        dynamic_actions_list.append((feat, action))
        
        with st.expander(f"Recommendation for {feat.upper()}"):
            st.write(f"**Why this?** This factor has the highest influence on this customer's risk profile.")
            st.info(action)

    # --- 5. PDF DOWNLOAD ---
    pdf_bytes = create_pdf(prediction, probability, dynamic_actions_list)
    st.download_button(
        label="📥 Download Strategy Report (PDF)",
        data=pdf_bytes,
        file_name="customer_retention_plan.pdf",
        mime="application/pdf"
    )

    # Static high-priority rule
    if input_df['CustServ Calls'].iloc[0] >= 4:
        st.warning("⚠️ **CRITICAL FLAG:** This customer has exceeded the 4-call service threshold. Immediate outreach required regardless of other metrics.")