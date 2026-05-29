import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------
# PAGE CONFIGURATION & ENTERPRISE THEME
# -------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Insurance Cross-Sell AI Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom injection of metric layouts and background cleanups
st.markdown("""
<style>
    /* Executive Card Layouts */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f5;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value-positive {
        font-size: 26px;
        font-weight: bold;
        color: #2b9348;
    }
    .metric-value-negative {
        font-size: 26px;
        font-weight: bold;
        color: #d90429;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #adb5bd;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# PIPELINE DATA INFRASTRUCTURE LOAD
# -------------------------------------------------------------
@st.cache_resource
def load_production_pipeline():
    try:
        # Load unified preprocessing objects & finalized model architecture
        payload = joblib.load('model_ins3_prod_files.pkl')
        return payload
    except FileNotFoundError:
        return None

pipeline = load_production_pipeline()

if pipeline is None:
    st.error("### 🛠️ Strategic Asset Missing")
    st.info("The mandatory production core file (`model_ins3_prod_files.pkl`) was not found in the root directory. Please deploy the artifacts to initialize.")
    st.stop()

# Deconstruct serialized payload with correct dictionary keys
model = pipeline['model']
OHE = pipeline['OHE']
Scaler = pipeline['Scaler']
model_cols = pipeline['model_columns']

# -------------------------------------------------------------
# BRANDING HEADER
# -------------------------------------------------------------
st.title("🚗 Predictive Insurance Cross-Sell Matrix")
st.markdown("An advanced pipeline optimized for predicting user conversion propensity for health-to-vehicle policy expansion.")
st.markdown("---")

# Tabbed Core Navigation Workspace
tab_scoring, tab_analytics = st.tabs(["🔮 Real-Time Customer Scoring", "📊 Distribution & Feature Analytics"])

# =============================================================
# TAB 1: ADVANCED MACHINE LEARNING PROPENSITY INTERFACE
# =============================================================
with tab_scoring:
    st.subheader("Customer Operational Parameter Setup")
    st.markdown("调整或者输入目标客户指标以实时计算转化率 (Adjust or input objective metrics to score target customer profile).")
    
    # 3-Column Advanced Sizing Matrix
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### **🧑 Demographic Identifiers**")
        gender = st.radio("Gender Representation", options=["Male", "Female"], horizontal=True)
        age = st.slider("Customer Structural Age (Years)", min_value=18, max_value=90, value=38)
        region_code_raw = st.number_input("Regional Boundary Core ID (Region_Code)", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
        region_code = str(region_code_raw)

    with col2:
        st.markdown("##### **🚘 Asset & Operational History**")
        vehicle_age = st.selectbox("Vehicle Operational Lifecycle Age", options=["< 1 Year", "1-2 Year", "> 2 Years"], index=1)
        vehicle_damage = st.radio("Prior Verified Vehicle Damage Record?", options=["Yes", "No"], horizontal=True)
        vintage = st.slider("Customer Tenure Term (Vintage Days)", min_value=0, max_value=350, value=142)

    with col3:
        st.markdown("##### **📄 Active Underwriting & Policy Specs**")
        driving_license = st.selectbox("Motor License Validation Status", options=["No Active License (0)", "Verified Holder (1)"], index=1)
        driving_license_val = 1 if "Verified Holder" in driving_license else 0
        
        previously_insured = st.selectbox("Active Vehicle Cover History", options=["Uninsured Asset (0)", "Existing External Cover (1)"], index=0)
        previously_insured_val = 1 if "Existing External Cover" in previously_insured else 0
        
        annual_premium = st.number_input("Calculated Annual Premium Gross (INR)", min_value=0.0, value=32640.0, step=500.0)
        sales_channel_raw = st.number_input("Policy Distribution Channel Token", min_value=0.0, max_value=300.0, value=26.0, step=1.0)
        policy_sales_channel = str(sales_channel_raw)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Large High-Impact Evaluation Callout Trigger
    if st.button("⚡ Execute High-Fidelity Predictive Inference", use_container_width=True, type="primary"):
        
        # Build vector framework matrix matching original train format
        raw_payload = {
            'Gender': gender,
            'Age': age,
            'Driving_License': driving_license_val,
    
    
        
