import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration for a clean, modern UI
st.set_page_config(
    page_title="Vehicle Insurance Cross-Sell Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom minimal CSS styling to match clean modern UI standards
st.markdown("""
<style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Application Heading & Description
st.title("🚗 Vehicle Insurance Cross-Sell Analyzer")
st.markdown("Predict customer intent and explore demographic behavior for optimizing cross-selling strategies.")

# Load production components safely
@st.cache_resource
def load_model_pipeline():
    try:
        artifacts = joblib.load('model_ins3_prod_files.pkl')
        return artifacts
    except FileNotFoundError:
        return None

artifacts = load_model_pipeline()

if artifacts is None:
    st.error("### ❌ Configuration Error")
    st.warning("The required production pipeline file (`model_ins2_prod_files.pkl`) was not found in the current folder. Please make sure the notebook training execution finished completely and saved the file.")
    st.stop()

# Unpack artifacts
model = artifacts['model']
OHE = artifacts['cat_encode']
Scaler = artifacts['num_encode']
model_cols = artifacts['columns']

# Create Top-level Tabs similar to your friend's structure
tab_predict, tab_insights = st.tabs(["🔮 Single Customer Prediction", "📊 Model Configuration & Insights"])

with tab_predict:
    st.markdown("### Feature Configuration Engine")
    st.info("Modify customer features across the columns below to generate real-time propensity scores.")
    
    # 3-Column functional setup layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### **Demographics**")
        gender = st.radio("Gender Profile", options=["Male", "Female"], horizontal=True)
        age = st.slider("Customer Age", min_value=18, max_value=100, value=35)
        region_code_input = st.number_input("Region Area Code ID", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
        region_code = str(region_code_input)

    with col2:
        st.markdown("#### **Vehicle Attributes**")
        vehicle_age = st.selectbox("Vehicle Age Class", options=["< 1 Year", "1-2 Year", "> 2 Years"], index=1)
        vehicle_damage = st.radio("Prior Vehicle Damage?", options=["Yes", "No"], horizontal=True)
        vintage = st.slider("Customer Tenure (Vintage Days)", min_value=0, max_value=500, value=150)

    with col3:
        st.markdown("#### **Policy Context**")
        driving_license = st.selectbox("Driving License Status", options=["No License (0)", "Has Active License (1)"], index=1)
        driving_license_val = 1 if "Has Active License" in driving_license else 0
        
        previously_insured = st.selectbox("Previously Insured Profile", options=["Not Insured (0)", "Already Covered (1)"], index=0)
        previously_insured_val = 1 if "Already Covered" in previously_insured else 0
        
        annual_premium = st.number_input("Annual Premium Investment Amount", min_value=0.0, value=35000.0, step=1000.0)
        policy_sales_channel_input = st.number_input("Policy Sales Channel ID", min_value=0.0, max_value=300.0, value=26.0, step=1.0)
        policy_sales_channel = str(policy_sales_channel_input)

    st.markdown("---")
    
    # Action block trigger
    if st.button("🚀 Calculate Customer Cross-Sell Propensity", use_container_width=True):
        
        # Structure payload matching training frame format
        raw_data = {
            'Gender': [gender],
            'Age': [age],
            'Driving_License': [driving_license_val],
            'Region_Code': [region_code],
            'Previously_Insured': [previously_insured_val],
            'Vehicle_Age': [vehicle_age],
            'Vehicle_Damage': [vehicle_damage],
            'Annual_Premium': [annual_premium],
            'Policy_Sales_Channel': [policy_sales_channel],
            'Vintage': [vintage]
        }
        
        input_df = pd.DataFrame(raw_data)
        
        # Pipeline Transformations
        cat_cols = ['Gender', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
        cat_df = input_df[cat_cols]
        
        try:
            # Categorical encoding
            encoded_cat = OHE.transform(cat_df)
            encoded_cat_df = pd.DataFrame(encoded_cat, columns=OHE.get_feature_names_out(cat_cols))
            
            # Numerical standard scaling
            num_cols = ['Age', 'Annual_Premium', 'Vintage']
            num_df = input_df[num_cols]
            scaled_num = Scaler.transform(num_df)
            scaled_num_df = pd.DataFrame(scaled_num, columns=num_cols)
            
            # Structural alignment
            bin_df = input_df[['Driving_License', 'Previously_Insured']].reset_index(drop=True)
            final_input_trans = pd.concat([encoded_cat_df, scaled_num_df, bin_df], axis=1)
            final_input_trans = final_input_trans.reindex(columns=model_cols, fill_value=0.0)
            
            # Predict outcome decision and model probabilities
            prediction = model.predict(final_input_trans)[0]
            
            # Handle probabilistic outputs safely depending on the classifier type
            if hasattr(model, "predict_proba"):
                prob_scores = model.predict_proba(final_input_trans)[0]
                interest_probability = prob_scores[1] * 100
            else:
                interest_probability = 100.0 if prediction == 1 else 0.0

            # UI Metrics display container blocks
            st.markdown("### Execution Analytics Output")
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Predicted Intent Class</div>
                    <div class="metric-value">{"Interested (1)" if prediction == 1 else "Not Interested (0)"}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Positive Response Propensity Score</div>
                    <div class="metric-value">{interest_probability:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Final Status Notifications based on classification logic
            if prediction == 1:
                st.success(f"🎯 **Target Strategy Recommendation:** Highly likely conversion candidate ({interest_probability:.1f}% intent certainty score). Deploy target outreach immediately.")
            else:
                st.warning(f"⚡ **Target Strategy Recommendation:** Unlikely candidate or cold outreach prospect ({interest_probability:.1f}% confidence score). Optimization advised.")
                
        except Exception as err:
            st.error(f"Execution Error during array pipeline translation: {err}")

with tab_insights:
    st.markdown("### Pipeline Deployment Insights")
    st.markdown("Verify operational parameters and feature columns generated directly from the model training file structure below.")
    
    # Showcase structural dimensions
    meta1, meta2 = st.columns(2)
    with meta1:
        st.metric(label="Total Features Expected", value=len(model_cols))
    with meta2:
        st.metric(label="Inference Class Type", value=type(model).__name__)
        
    with st.expander("🔍 Inspect Full Registered Pipeline Columns"):
        st.write(model_cols)
