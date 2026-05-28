import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Set Page Configuration
st.set_page_config(
    page_title="Vehicle Insurance Cross-Sell Predictor",
    page_icon="🚗",
    layout="centered"
)

# 2. Add Stylized Headers & Information banner
st.title("🚗 Vehicle Insurance Cross-Sell Predictor")
st.markdown("""
This production application utilizes a trained **K-Nearest Neighbors (KNN)** pipeline to evaluate 
the propensity of an existing health insurance consumer to purchase additional automobile protection coverage.
""")

# 3. Load the Saved Production Artifact safely
@st.cache_resource
def load_pipeline():
    # Make sure 'model_insurance_prod_files.pkl' is uploaded in the same GitHub repository path
    pipeline_data = joblib.load('model_insurance_prod_files.pkl')
    return pipeline_data

try:
    artifacts = load_pipeline()
    model = artifacts['model']
    num_scale = artifacts['num_encode']
    cat_encode = artifacts['cat_encode']
    st.success("🤖 Machine Learning Predictive Pipeline Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts. Ensure 'model_insurance_prod_files.pkl' is present in the repository directory. Error details: {e}")
    st.stop()

# 4. Construct Layout Input Sections for your 10 distinct features
st.subheader("📋 Enter Customer Profile Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35)
    driving_license = st.selectbox("Has Driving License?", options=["Yes", "No"])
    previously_insured = st.selectbox("Previously Insured Vehicles?", options=["Yes", "No"])
    vehicle_damage = st.selectbox("Ever Had Vehicle Damage?", options=["Yes", "No"])

with col2:
    vehicle_age = st.selectbox("Vehicle Age Group", options=["< 1 Year", "1-2 Year", "> 2 Years"])
    annual_premium = st.number_input("Annual Premium Paid (INR)", min_value=0.0, max_value=600000.0, value=35000.0, step=500.0)
    vintage = st.number_input("Customer Association Lifecycle Time (Vintage Days)", min_value=0, max_value=365, value=150)
    
    # Text input configuration matches your notebook's numeric categorization strings (.astype(str))
    region_code = st.text_input("Region Code Identification", value="28.0")
    policy_sales_channel = st.text_input("Policy Sales Channel Code", value="26.0")

# 5. Transform user selections to raw matching training dataframe format upon execution click
if st.button("🔮 Calculate Cross-Sell Propensity Prediction", type="primary"):
    
    # Map interactive visual values to binary integers used inside your raw dataset rows
    license_val = 1 if driving_license == "Yes" else 0
    insured_val = 1 if previously_insured == "Yes" else 0
    
    # Structure features into matching order dataframe layout (excluding dropped 'id' column)
    raw_input_dict = {
        'Gender': [gender],
        'Age': [age],
        'Driving_License': [license_val],
        'Region_Code': [str(float(region_code)) if region_code.replace('.', '', 1).isdigit() else str(region_code)],
        'Previously_Insured': [insured_val],
        'Vehicle_Age': [vehicle_age],
        'Vehicle_Damage': [vehicle_damage],
        'Annual_Premium': [float(annual_premium)],
        'Policy_Sales_Channel': [str(float(policy_sales_channel)) if policy_sales_channel.replace('.', '', 1).isdigit() else str(policy_sales_channel)],
        'Vintage': [int(vintage)]
    }
    
    input_df = pd.DataFrame(raw_input_dict)
    
    # Apply type conversions explicitly matching original preprocessing instructions
    input_df['Region_Code'] = input_df['Region_Code'].astype(str)
    input_df['Policy_Sales_Channel'] = input_df['Policy_Sales_Channel'].astype(str)
    
    # 6. Apply Parallel Processing Transformation Streams
    num_cols = ['Age', 'Annual_Premium', 'Vintage']
    cat_cols = ['Gender', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
    
    try:
        # Scale continuous metrics using the fitted training MinMaxScaler bounds
        transformed_num = num_scale.transform(input_df[num_cols])
        
        # Vectorize text features using fitted OneHotEncoder columns
        transformed_cat = cat_encode.transform(input_df[cat_cols])
        
        # Horizontal stack combination exactly mirrors your notebook engineering logic
        if hasattr(transformed_cat, "toarray"):
            transformed_cat = transformed_cat.toarray()
            
        final_processed_features = np.hstack((transformed_num, transformed_cat))
        
        # 7. Execute Prediction Metric Block
        prediction = model.predict(final_processed_features)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.markdown("### 🔥 Propensity Outcome: **High Conversion Probability!**")
            st.info("🎯 **Recommendation:** This consumer is flagged as highly likely to cross-purchase vehicle insurance coverage. Pass this lead to the active target conversion registry lists.")
        else:
            st.markdown("### 📊 Propensity Outcome: **Low Conversion Probability**")
            st.warning("💤 **Recommendation:** Consumer exhibits low interest markers for automobile plans. Retain default automated generic billing templates only.")
            
    except Exception as transformation_error:
        st.error(f"Transformation Error: Ensure input category keys precisely align with parameters extracted during training. Error details: {transformation_error}")