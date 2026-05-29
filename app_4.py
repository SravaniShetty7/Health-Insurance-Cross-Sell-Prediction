import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------------
# PAGE CONFIGURATION & METRIC LAYOUT STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Insurance Cross-Sell Matrix",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Native CSS injection for premium card look and error insulation
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #eef2f5;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value-positive {
        font-size: 28px;
        font-weight: bold;
        color: #2b9348;
    }
    .metric-value-negative {
        font-size: 28px;
        font-weight: bold;
        color: #d90429;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #adb5bd;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DEFENSIVE PIPELINE LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_production_pipeline():
    try:
        payload = joblib.load('model_ins2_prod_files.pkl')
        return payload
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to unpickle model artifacts: {e}")
        return None

pipeline = load_production_pipeline()

if pipeline is None:
    st.error("### 🛠️ Production Pipeline Unloaded")
    st.warning("`model_ins2_prod_files.pkl` could not be initialized or was not found in the current working directory.")
    st.info("💡 **Fix:** Run the last cells of your `PROJECT.ipynb` notebook to export your model, OHE encoder, Scaler, and list of training columns into the pickle package.")
    st.stop()

# Deconstruct payload
try:
    model = pipeline['model']
    OHE = pipeline['cat_encode']
    Scaler = pipeline['num_encode']
    model_cols = pipeline['model_columns']
except KeyError as ke:
    st.error(f"Missing key dictionary mapping in your exported file: {ke}")
    st.stop()

# -------------------------------------------------------------
# APPLICATION HEADER UI
# -------------------------------------------------------------
st.title("🚗 Predictive Insurance Cross-Sell Matrix")
st.markdown("An advanced predictive analytics cockpit mapping customer demographic metrics to conversion propensities.")
st.markdown("---")

tab_scoring, tab_analytics = st.tabs(["🔮 Real-Time Customer Scoring", "📊 Distribution & Feature Analytics"])

# =============================================================
# TAB 1: MACHINE LEARNING PROPENSITY TESTING INTERFACE
# =============================================================
with tab_scoring:
    st.subheader("Customer Parameter Workspace")
    st.markdown("Adjust configuration metrics below to generate instant model propensity decisions.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### **🧑 Demographic Profile**")
        gender = st.radio("Gender Representation", options=["Male", "Female"], horizontal=True)
        age = st.slider("Customer Structural Age (Years)", min_value=18, max_value=90, value=38)
        region_code_raw = st.number_input("Regional Boundary Core ID", min_value=0.0, max_value=100.0, value=28.0, step=1.0)

    with col2:
        st.markdown("##### **🚘 Asset & Operational History**")
        vehicle_age = st.selectbox("Vehicle Lifecycle Categorization", options=["< 1 Year", "1-2 Year", "> 2 Years"], index=1)
        vehicle_damage = st.radio("Prior Verified Vehicle Damage?", options=["Yes", "No"], horizontal=True)
        vintage = st.slider("Customer Tenure Term (Vintage Days)", min_value=0, max_value=350, value=142)

    with col3:
        st.markdown("##### **📄 Policy Context**")
        driving_license = st.selectbox("Motor License Validation Status", options=["No Active License (0)", "Verified Holder (1)"], index=1)
        driving_license_val = 1 if "Verified Holder" in driving_license else 0
        
        previously_insured = st.selectbox("Active Vehicle Cover Status", options=["Uninsured Asset (0)", "Existing External Cover (1)"], index=0)
        previously_insured_val = 1 if "Existing External Cover" in previously_insured else 0
        
        annual_premium = st.number_input("Calculated Annual Premium (INR)", min_value=0.0, value=32640.0, step=500.0)
        sales_channel_raw = st.number_input("Policy Distribution Channel ID", min_value=0.0, max_value=300.0, value=26.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("⚡ Execute Robust Predictive Inference", use_container_width=True, type="primary"):
        
        # -------------------------------------------------------------
        # STRICT DATATYPE MATCHING & ALIGNMENT PIPELINE
        # -------------------------------------------------------------
        try:
            # Recreate raw feature dataframe using explicit dictionary formatting
            raw_payload = {
                'Gender': [gender],
                'Age': [float(age)],
                'Driving_License': [int(driving_license_val)],
                'Region_Code': [float(region_code_raw)], 
                'Previously_Insured': [int(previously_insured_val)],
                'Vehicle_Age': [vehicle_age],
                'Vehicle_Damage': [vehicle_damage],
                'Annual_Premium': [float(annual_premium)],
                'Policy_Sales_Channel': [float(sales_channel_raw)],
                'Vintage': [float(vintage)]
            }
            
            input_frame = pd.DataFrame(raw_payload)
            
            # Match data mutations used exactly inside training cells (e.g., casting float IDs to string labels)
            input_frame['Region_Code'] = input_frame['Region_Code'].astype(str)
            input_frame['Policy_Sales_Channel'] = input_frame['Policy_Sales_Channel'].astype(str)
            
            # --- TRANSFORM STEP 1: ONE HOT ENCODING ---
            cat_features = ['Gender', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
            
            # Wrap in clean pandas structures to handle OHE transform output types safely
            encoded_cats_raw = OHE.transform(input_frame[cat_features])
            if hasattr(encoded_cats_raw, "toarray"):
                encoded_cats_raw = encoded_cats_raw.toarray()
                
            encoded_cats_df = pd.DataFrame(encoded_cats_raw, columns=OHE.get_feature_names_out(cat_features))
            
            # --- TRANSFORM STEP 2: SCALING ---
            num_features = ['Age', 'Annual_Premium', 'Vintage']
            scaled_nums_df = pd.DataFrame(Scaler.transform(input_frame[num_features]), columns=num_features)
            
            # --- TRANSFORM STEP 3: PIPELINE SYNCHRONIZATION ---
            binary_df = input_frame[['Driving_License', 'Previously_Insured']].reset_index(drop=True)
            fully_transformed_vector = pd.concat([encoded_cats_df, scaled_nums_df, binary_df], axis=1)
            
            # Defensive Column Padding: Ensure column dimensions exactly mirror train state spaces
            fully_transformed_vector = fully_transformed_vector.reindex(columns=model_cols, fill_value=0.0)
            
            # --- INFERENCE STEP ---
            predicted_class = int(model.predict(fully_transformed_vector)[0])
            
            # Safely capture response probabilities
            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(fully_transformed_vector)[0]
                conversion_probability = float(prob_array[1] * 100)
            else:
                conversion_probability = 100.0 if predicted_class == 1 else 0.0
                
            # -------------------------------------------------------------
            # UI METRICS CARDS GENERATION
            # -------------------------------------------------------------
            st.markdown("#### **📊 Operational Pipeline Inference Results**")
            card_col1, card_col2 = st.columns(2)
            
            with card_col1:
                class_style = "metric-value-positive" if predicted_class == 1 else "metric-value-negative"
                class_label = "CONVERSION TARGET (1)" if predicted_class == 1 else "UNINTERESTED PROSPECT (0)"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Target Classification Output</div>
                    <div class="{class_style}">{class_label}</div>
                    <div class="metric-subtitle">Optimized Binary Matrix Threshold Output</div>
                </div>
                """, unsafe_allow_html=True)
                
            with card_col2:
                prob_style = "metric-value-positive" if conversion_probability >= 50.0 else "metric-value-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Calculated Conversion Propensity Score</div>
                    <div class="{prob_style}">{conversion_probability:.2f}%</div>
                    <div class="metric-subtitle">Continuous Target Cross-Sell Score Range</div>
                </div>
                """, unsafe_allow_html=True)

            # Final Tactical Feedback Flags
            if predicted_class == 1:
                st.success(f"🎯 **High-Priority Conversion Opportunity:** Profile matches conversion parameters ({conversion_probability:.1f}% confidence score). Deploy aggressive targeted direct outbound action.")
            else:
                st.warning(f"📉 **Low-Yield Account Identified:** Conversion probability is sub-optimal ({conversion_probability:.1f}% score). Conserve standard direct marketing capital or route to lower priority status.")
                
        except ValueError as val_err:
            st.error("### ⚠️ Feature Alignment Error")
            st.markdown(f"**Details:** `{val_err}`")
            st.info("This normally happens if an unexpected user configuration input maps into the OneHotEncoder pipeline. Please ensure categories match your initial training parameters.")
        except Exception as general_err:
            st.error(f"### 🛑 Pipeline Execution Interrupted")
            st.markdown(f"**Error Runtime Signature:** `{general_err}`")

# =============================================================
# TAB 2: METRICS & COLUMN AUDIT MATRIX
# =============================================================
with tab_analytics:
    st.subheader("Data Dimensions & Feature Schemas")
    st.markdown("Audit the column shape configurations received directly from the model pickle file below.")
    
    sub1, sub2, sub3 = st.columns(3)
    with sub1:
        st.metric(label="Expected Column Feature Count", value=f"{len(model_cols)} Columns")
    with sub2:
        st.metric(label="Model Estimator Class Type", value=type(model).__name__)
    with sub3:
        st.metric(label="Core Pipeline Initialization", value="Successful & Safe")
        
    st.markdown("---")
    st.markdown("#### Historical Benchmark Validation Target Frequencies")
    
    mock_distribution_data = pd.DataFrame({
        'Response Class': ['Not Interested (0)', 'Interested (1)'],
        'Base Benchmark Share (%)': [87.8, 12.2]
    }).set_index('Response Class')
    
    st.bar_chart(mock_distribution_data)
    
    with st.expander("🔍 Advanced Inspection: Expected Pipeline Array Alignment Schema"):
        st.dataframe(pd.Series(model_cols, name="Registered Input Vector Features Array Addresses"))
  
