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
    st.info("The mandatory production core file (`model_ins2_prod_files.pkl`) was not found in the root directory. Please deploy the artifacts to initialize.")
    st.stop()

# Deconstruct serialized payload
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
            'Region_Code': region_code,
            'Previously_Insured': previously_insured_val,
            'Vehicle_Age': vehicle_age,
            'Vehicle_Damage': vehicle_damage,
            'Annual_Premium': annual_premium,
            'Policy_Sales_Channel': policy_sales_channel,
            'Vintage': vintage
        }
        
        input_frame = pd.DataFrame(raw_payload,index=[0])
        
        try:
        # 1. Transform ONLY the columns the One-Hot Encoder expects
        cat_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        encoded_cats = OHE.transform(input_frame[cat_features])
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=OHE.get_feature_names_out(cat_features))
        
        # 2. Rescale Continuous metrics via Scaler (Make sure numbers match your notebook scale)
        num_features = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
        scaled_nums = Scaler.transform(input_frame[num_features])
        scaled_nums_df = pd.DataFrame(scaled_nums, columns=num_features)
        
        # 3. Combine them back together along with the raw binary flags
        processed_df = pd.concat([
            scaled_nums_df, 
            encoded_cats_df, 
            input_frame[['Driving_License', 'Previously_Insured']].reset_index(drop=True)
        ], axis=1)
        
        # 4. Reindex to force match all 218 model columns in the exact original order
        final_model_input = processed_df.reindex(columns=model_cols, fill_value=0)
        
        # 5. Execute Prediction & Probabilities
        prediction = model.predict(final_model_input)[0]
        probability = model.predict_proba(final_model_input)[0][1]
        
        # 6. UI Output Presentation
        st.markdown("---")
        if prediction == 1:
            st.success(f"🎉 **High Propensity!** This customer is highly likely to purchase Vehicle Insurance. (Confidence Score: {probability:.1%})")
            st.balloons()
        else:
            st.warning(f"🛑 **Low Propensity.** This customer is unlikely to opt for vehicle cross-selling. (Confidence Score: {1 - probability:.1%})")

    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
            # Enforce tracking dimensions matches input requirement structure
            fully_transformed_vector = fully_transformed_vector.reindex(columns=model_cols, fill_value=0.0)
            
            # Compute operational outcome metrics
            predicted_class = model.predict(fully_transformed_vector)[0]
            
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(fully_transformed_vector)[0]
                conversion_probability = prob_array[1] * 100
            else:
                conversion_probability = 100.0 if predicted_class == 1 else 0.0
                
            # DISPLAY CUSTOM SYSTEM METRICS CARDS
            st.markdown("#### **📊 Operational Pipeline Inference Results**")
            card_col1, card_col2 = st.columns(2)
            
            with card_col1:
                class_style = "metric-value-positive" if predicted_class == 1 else "metric-value-negative"
                class_label = "POSITIVE INTEREST (1)" if predicted_class == 1 else "NEGATIVE DESIRE (0)"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Target Classification Output</div>
                    <div class="{class_style}">{class_label}</div>
                    <div class="metric-subtitle">Optimized Binary Decision Output Threshold</div>
                </div>
                """, unsafe_allow_html=True)
                
            with card_col2:
                prob_style = "metric-value-positive" if conversion_probability >= 50.0 else "metric-value-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Calculated Propensity Probability</div>
                    <div class="{prob_style}">{conversion_probability:.2f}%</div>
                    <div class="metric-subtitle">Continuous Target Cross-Sell Score Distribution</div>
                </div>
                """, unsafe_allow_html=True)

            # Execution Action Alert Rules
            if predicted_class == 1:
                st.success(f"🎯 **High-Priority Target Candidate Confirmed:** Propensity profile meets confidence thresholds ({conversion_probability:.1f}% score). Recommended action: Prioritize for standard retention and outbound outreach campaigns.")
            else:
                st.warning(f"📉 **Low-Yield Target Prospect Identified:** Customer parameters indicate suboptimal alignment with conversion patterns ({conversion_probability:.1f}% score). Recommended action: Conserve marketing resources or pair with premium discount options.")
                
        except Exception as pipeline_fault:
            st.error(f"Operational pipeline execution fault: {pipeline_fault}")
            st.info("Verify numerical format values match production settings.")

# =============================================================
# TAB 2: EXPLORATION INSIGHTS & DISTRIBUTION PROFILE
# =============================================================
with tab_analytics:
    st.subheader("Data Profiles & Feature Spaces")
    st.markdown("Review downstream engineering metrics, distribution checks, and feature array shapes mapped directly from training.")
    
    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        st.metric(label="Model Matrix Feature Vector Shapes", value=f"{len(model_cols)} Columns")
    with sub2:
        st.metric(label="Calculated Model Architecture Target", value=type(model).__name__)
    with sub3:
        st.metric(label="Core Pipeline State", value="Fitted & Hot-Loaded")
        
    st.markdown("---")
    st.markdown("#### Expected Target Distribution Check (Reference Metrics)")
    
    # Build beautiful analytical benchmark visual comparisons for professional tracking
    mock_distribution_data = pd.DataFrame({
        'Response Outcome Class': ['Not Interested (0)', 'Interested (1)'],
        'Historical Base Share (%)': [87.8, 12.2]
    }).set_index('Response Outcome Class')
    
    st.bar_chart(mock_distribution_data)
    
    with st.expander("🔍 Advanced Debugging: View Encoded Matrix Array Schema"):
        st.dataframe(pd.Series(model_cols, name="Registered Input Vector Columns Address Key"))
