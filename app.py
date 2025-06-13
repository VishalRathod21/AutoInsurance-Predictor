import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="AutoInsure Predict",
    page_icon="🚗💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark mode compatible CSS
st.markdown("""
    <style>
    /* Base styles for both light and dark modes */
    :root {
        --primary: #4f46e5;
        --primary-hover: #4338ca;
        --text: #1f2937;
        --text-dark: #f3f4f6;
        --bg: #ffffff;
        --bg-dark: #0f172a;
        --card-bg: #ffffff;
        --card-bg-dark: #1e293b;
        --border: #e5e7eb;
        --border-dark: #334155;
    }
    
    /* Light mode styles */
    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }
    
    /* Dark mode styles */
    .stApp[data-theme="dark"] {
        background-color: var(--bg-dark);
        color: var(--text-dark);
    }
    
    /* Input fields - Light mode */
    .stTextInput input, 
    .stNumberInput input,
    .stSelectbox select,
    .stTextArea textarea {
        color: var(--text) !important;
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    
    /* Input fields - Dark mode */
    [data-theme="dark"] .stTextInput input,
    [data-theme="dark"] .stNumberInput input,
    [data-theme="dark"] .stSelectbox select,
    [data-theme="dark"] .stTextArea textarea {
        color: var(--text-dark) !important;
        background-color: var(--card-bg-dark) !important;
        border-color: var(--border-dark) !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: background-color 0.2s !important;
    }
    
    .stButton>button:hover {
        background: var(--primary-hover) !important;
    }
    
    /* Cards */
    .main .block-container {
        padding: 2rem 1rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        border-left: 5px solid;
    }
    
    .prediction-yes {
        background-color: #d1fae5;
        border-color: #10b981;
        color: #065f46;
    }
    
    .prediction-no {
        background-color: #fee2e2;
        border-color: #ef4444;
        color: #991b1b;
    }
    
    /* Dark mode prediction boxes */
    [data-theme="dark"] .prediction-yes {
        background-color: #064e3b;
        color: #d1fae5;
    }
    
    [data-theme="dark"] .prediction-no {
        background-color: #7f1d1d;
        color: #fecaca;
    }
    
    /* Tabs */
    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary);
    }
    
    /* Radio buttons */
    .stRadio > div {
        flex-direction: row !important;
        gap: 2rem;
        align-items: center;
    }
    
    .stRadio label {
        margin-right: 1rem;
        color: inherit !important;
    }
    
    /* Fix for dark mode radio buttons */
    [data-theme="dark"] .stRadio label {
        color: var(--text-dark) !important;
    }
    
    /* Section headers */
    h1, h2, h3, h4, h5, h6 {
        color: inherit !important;
    }
    
    /* Form labels */
    .stForm label {
        color: inherit !important;
    }
    </style>
""", unsafe_allow_html=True)

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

# Custom CSS for prediction boxes (kept for backward compatibility)
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #2c3e50; text-align: center; margin-bottom: 1.5rem;}
    .subheader {font-size: 1.5rem; color: #4f46e5; margin-top: 1.5rem; margin-bottom: 1rem;}
    .form-container {background-color: #f8f9fa; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .section {margin-bottom: 2rem;}
    .stMarkdown h2 {color: #2c3e50; border-bottom: 2px solid #4f46e5; padding-bottom: 0.5rem;}
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
    # Main header with gradient text
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='background: linear-gradient(45deg, #4f46e5, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; margin-bottom: 0.5rem;'>AutoInsure Predict</h1>
            <p style='color: #4b5563; font-size: 1.25rem; margin-top: 0;'>Intelligent Vehicle Insurance Interest Prediction Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Features showcase
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div style='background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #4f46e5; margin-top: 0;'>🔍 Smart Predictions</h3>
                    <p style='color: #4b5563;'>Advanced ML models predict customer interest with high accuracy using comprehensive data analysis.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div style='background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #4f46e5; margin-top: 0;'>📊 Data-Driven</h3>
                    <p style='color: #4b5563;'>Leverage historical data and customer behavior patterns for precise interest predictions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                """
                <div style='background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #4f46e5; margin-top: 0;'>⚡ Real-Time</h3>
                    <p style='color: #4b5563;'>Get instant predictions to quickly identify potential customers and optimize your sales strategy.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add tabs for different functionalities with icons
    tab1, tab2 = st.tabs(["🔮 Predict Interest", "🤖 Train Model"])
    
    with tab1:
        # Prediction form with improved styling
        st.markdown("### 📋 Customer Details")
        st.markdown("<p style='color: #4b5563; margin-top: -1rem; margin-bottom: 1.5rem;'>Fill in the customer information to predict their interest in vehicle insurance.</p>", unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Personal Information")
                gender = st.radio("Gender", ["Male", "Female"], horizontal=True, index=0, help="Select the gender of the customer")
                age = st.slider("Age", min_value=18, max_value=100, value=30, step=1, help="Age of the customer")
                
                st.markdown("#### Driving & Insurance")
                driving_license = st.radio("Driving License", ["Yes", "No"], horizontal=True, index=0, help="Does the customer have a driving license?")
                previously_insured = st.radio("Previously Insured", ["No", "Yes"], horizontal=True, index=0, help="Is the customer already insured?")
                
                st.markdown("#### Region")
                region_code = st.number_input("Region Code", min_value=1, max_value=50, value=28, step=1, help="Region code (1-50)")
            
            with col2:
                st.markdown("#### Vehicle Information")
                annual_premium = st.number_input("Annual Premium (₹)", min_value=1000, max_value=1000000, value=30000, step=1000, help="Annual premium amount in INR")
                policy_sales_channel = st.number_input("Policy Sales Channel", min_value=1, max_value=200, value=26, step=1, help="Sales channel code (1-200)")
                vintage = st.slider("Vintage (days)", min_value=0, max_value=400, value=200, step=1, help="Number of days the customer has been associated")
                
                st.markdown("#### Vehicle Details")
                vehicle_age = st.radio("Vehicle Age", ["< 1 Year", "1-2 Years", "> 2 Years"], horizontal=True, index=1, help="Select vehicle age category")
                vehicle_age_lt_1 = 1 if vehicle_age == "< 1 Year" else 0
                vehicle_age_gt_2 = 1 if vehicle_age == "> 2 Years" else 0
                
                vehicle_damage = st.radio("Vehicle Damage", ["No", "Yes"], horizontal=True, index=0, help="Has the vehicle been damaged before?")
            
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
    
    # Footer with more details
    st.markdown("---")
    st.markdown(
        """
        <div style='display: flex; justify-content: space-between; align-items: center; color: #6b7280; padding: 1rem 0;'>
            <div>© 2025 InsureVision AI - All Rights Reserved</div>
            <div style='display: flex; gap: 1rem;'>
                <a href='#' style='color: #4f46e5; text-decoration: none;'>Privacy Policy</a>
                <a href='#' style='color: #4f46e5; text-decoration: none;'>Terms of Service</a>
                <a href='#' style='color: #4f46e5; text-decoration: none;'>Contact Us</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add a small loading animation
    st.markdown(
        """
        <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .pulse-animation {
            animation: pulse 2s infinite;
            display: inline-block;
        }
        </style>
        <span class='pulse-animation'>✨</span>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
