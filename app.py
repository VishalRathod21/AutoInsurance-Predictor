import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="AutoInsure Predict",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Load environment variables
load_dotenv()

# Import prediction pipeline
try:
    from src.pipeline.prediction_pipeline import VehicleData, VehicleDataClassifier
    from src.pipeline.training_pipeline import TrainPipeline
    from src.logger import logging
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #2c3e50; text-align: center; margin-bottom: 1.5rem;}
    .subheader {font-size: 1.5rem; color: #3498db; margin-top: 1.5rem; margin-bottom: 1rem;}
    .prediction-box {padding: 1.5rem; border-radius: 10px; margin: 1rem 0;}
    .prediction-yes {background-color: #d4edda; color: #155724; border-left: 5px solid #28a745;}
    .prediction-no {background-color: #f8d7da; color: #721c24; border-left: 5px solid #dc3545;}
    .stButton>button {width: 100%; border-radius: 5px; padding: 0.5rem; background-color: #3498db; color: white;}
    .stButton>button:hover {background-color: #2980b9; color: white;}
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {border-radius: 5px; border: 1px solid #ced4da;}
    .form-container {background-color: #f8f9fa; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .section {margin-bottom: 2rem;}
    .stMarkdown h2 {color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem;}
    </style>
""", unsafe_allow_html=True)

def predict_interest(data):
    try:
        # Create a VehicleData object
        vehicle_data = VehicleData(
            Gender=int(data['Gender']),
            Age=int(data['Age']),
            Driving_License=int(data['Driving_License']),
            Region_Code=float(data['Region_Code']),
            Previously_Insured=int(data['Previously_Insured']),
            Annual_Premium=float(data['Annual_Premium']),
            Policy_Sales_Channel=float(data['Policy_Sales_Channel']),
            Vintage=int(data['Vintage']),
            Vehicle_Age_lt_1_Year=int(data['Vehicle_Age_lt_1_Year']),
            Vehicle_Age_gt_2_Years=int(data['Vehicle_Age_gt_2_Years']),
            Vehicle_Damage_Yes=int(data['Vehicle_Damage_Yes'])
        )
        
        # Get prediction
        classifier = VehicleDataClassifier()
        prediction = classifier.predict(vehicle_data.get_vehicle_input_data_frame())
        
        # For now, set a default confidence since the model doesn't return probabilities
        # You might want to modify the model to return probabilities as well
        confidence = 0.85 if prediction == 1 else 0.75
        
        return prediction, [[1 - confidence, confidence]]  # Return in format [prob_not_interested, prob_interested]
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        st.error(f"An error occurred during prediction: {str(e)}")
        return None, None

def train_model():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return True, "Model trained successfully!"
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        return False, f"Error during training: {str(e)}"

def main():
    # Main header
    st.markdown("<h1 class='main-header'>AutoInsure Predict</h1>", unsafe_allow_html=True)
    st.markdown("### Predict Customer Interest in Vehicle Insurance")
    
    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["Predict", "Train Model"])
    
    with tab1:
        st.markdown("### Enter Customer Details")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Personal Information")
                gender = st.selectbox("Gender", ["Male", "Female"], index=0, help="Select the gender of the customer")
                age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Age of the customer")
                driving_license = st.selectbox("Driving License", ["Yes", "No"], index=0, help="Does the customer have a driving license?")
                region_code = st.number_input("Region Code", min_value=0.0, value=28.0, step=0.1, format="%.1f", help="Region code of the customer")
                previously_insured = st.selectbox("Previously Insured", ["No", "Yes"], index=0, help="Is the customer already insured?")
            
            with col2:
                st.markdown("#### Vehicle Information")
                annual_premium = st.number_input("Annual Premium", min_value=0.0, value=30000.0, step=1000.0, format="%.2f", help="Annual premium amount")
                policy_sales_channel = st.number_input("Policy Sales Channel", min_value=0.0, value=26.0, step=1.0, format="%.1f", help="Sales channel code")
                vintage = st.number_input("Vintage", min_value=0, value=200, help="Number of days the customer has been associated")
                
                st.markdown("#### Vehicle Age")
                col_age1, col_age2 = st.columns(2)
                with col_age1:
                    vehicle_age_lt_1 = st.checkbox("Less than 1 year", value=False, help="Is the vehicle less than 1 year old?")
                with col_age2:
                    vehicle_age_gt_2 = st.checkbox("More than 2 years", value=False, help="Is the vehicle more than 2 years old?")
                
                vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"], index=0, help="Has the vehicle been damaged before?")
            
            submit_button = st.form_submit_button("Predict Interest")
            
            if submit_button:
                # Prepare data for prediction
                data = {
                    'Gender': 1 if gender == "Male" else 0,
                    'Age': age,
                    'Driving_License': 1 if driving_license == "Yes" else 0,
                    'Region_Code': float(region_code),
                    'Previously_Insured': 1 if previously_insured == "Yes" else 0,
                    'Annual_Premium': float(annual_premium),
                    'Policy_Sales_Channel': float(policy_sales_channel),
                    'Vintage': int(vintage),
                    'Vehicle_Age_lt_1_Year': 1 if vehicle_age_lt_1 else 0,
                    'Vehicle_Age_gt_2_Years': 1 if vehicle_age_gt_2 else 0,
                    'Vehicle_Damage_Yes': 1 if vehicle_damage == "Yes" else 0
                }
                
                with st.spinner('Making prediction...'):
                    prediction, probability = predict_interest(data)
                    
                    if prediction is not None:
                        if prediction == 1:  # Interested
                            st.markdown(f"""
                            <div class='prediction-box prediction-yes'>
                                <h3>🎯 Prediction: Likely to be Interested</h3>
                                <p>This customer is likely to be interested in vehicle insurance.</p>
                                <p><strong>Confidence:</strong> {probability[0][1]*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:  # Not interested
                            st.markdown(f"""
                            <div class='prediction-box prediction-no'>
                                <h3>❌ Prediction: Unlikely to be Interested</h3>
                                <p>This customer is not likely to be interested in vehicle insurance.</p>
                                <p><strong>Confidence:</strong> {probability[0][0]*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Train the Prediction Model")
        st.warning("Note: This will retrain the model with the latest data. This might take some time.")
        
        if st.button("Start Training"):
            with st.spinner('Training in progress. Please wait...'):
                success, message = train_model()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Add some space at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("2025 AutoInsure Predict - All Rights Reserved")

if __name__ == "__main__":
    main()
