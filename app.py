import streamlit as st
import joblib
import pandas as pd
import warnings
import base64
import os

# Suppress warnings from scikit-learn version differences if any
warnings.filterwarnings("ignore")

# --- Page CONFIG ---
st.set_page_config(page_title="Flood Risk Model Predictor", page_icon="🌊", layout="wide")

def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(13, 17, 23, 0.85), rgba(13, 17, 23, 0.85)), url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

add_bg_from_local("src/bg_optim.jpg")

# --- Custom CSS for UI Aesthetics ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.8rem;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        padding-top: 20px;
    }

    .sub-header {
        text-align: center; 
        color: #a0aec0; 
        margin-bottom: 40px;
        font-size: 1.1rem;
    }

    .predict-box {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(145deg, #1a202c, #2d3748);
        box-shadow: 6px 6px 12px #0f131a, -6px -6px 12px #3b475a;
        text-align: center;
        margin: 10px;
        transition: transform 0.2s ease;
    }
    
    .predict-box:hover {
        transform: translateY(-5px);
    }

    .model-title {
        color: #cbd5e0;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #4facfe;
        text-shadow: 0 0 10px rgba(79, 172, 254, 0.4);
    }
    .risk-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-low {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    .risk-high {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(79, 172, 254, 0.75), rgba(0, 0, 0, 0));
        margin: 40px 0;
    }
    /* Custom Override: Only Sliders are Blue */
    .stSlider div[data-baseweb="slider"] {
        filter: hue-rotate(170deg) brightness(1.1);
    }

    /* Glassmorphism Animated Primary Button */
    button[kind="primary"] {
        background: rgba(79, 172, 254, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 15px 10px !important;
        min-height: 65px !important;
        letter-spacing: 1px !important;
        border-radius: 12px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; 
    }

    button[kind="primary"]:hover {
        background: rgba(79, 172, 254, 0.25) !important;
        border: 1px solid rgba(79, 172, 254, 0.8) !important;
        box-shadow: 0 15px 35px rgba(79, 172, 254, 0.4) !important;
        transform: translateY(-5px) scale(1.02) !important;
    }

    button[kind="primary"]:active {
        transform: translateY(2px) scale(0.98) !important;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource(show_spinner=False)
def load_models():
    lr = joblib.load('linear_regression_model.joblib')
    rf = joblib.load('random_forest_model.joblib')
    return lr, rf

with st.spinner("Initializing AI Models..."):
    lr_model, rf_model = load_models()

# --- App Structure ---
st.markdown("<div class='main-header'>🌊 Flood Risk Prediction & Model Interpeter</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Adjust the environmental and infrastructural factors below to simulate conditions and compare predictions using <b>Linear Regression</b> and <b>Random Forest</b> models.</div>", unsafe_allow_html=True)

# List of precise feature names as extracted from model
features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

st.subheader("📊 Define Environmental Inputs")

# Create a sleek layout for inputs using 4 columns
cols = st.columns(4)
input_data = {}

for idx, feature in enumerate(features):
    col = cols[idx % 4]
    with col:
        # Range is typically 0-15 or 0-20 for such scaled disaster datasets
        input_data[feature] = st.slider(f"{feature}", min_value=0, max_value=20, value=5, step=1)

st.markdown("<hr/>", unsafe_allow_html=True)

# Centered Predict Button
# The button width is controlled by this layout ratio. [1,5,1] makes the center block significantly wider.
pred_col1, pred_col2, pred_col3 = st.columns([5,4,5])
with pred_col2:
    predict_clicked = st.button("🚀 Analyze Risk & Generate Predictions", use_container_width=True, type="primary")

if predict_clicked:
    # Build dataframe for models in correct feature order
    input_df = pd.DataFrame([input_data])[features]
    
    with st.spinner("Models computing risk predictions..."):
        lr_pred = lr_model.predict(input_df)[0]
        rf_pred = rf_model.predict(input_df)[0]
        
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e2e8f0; margin-bottom: 20px;'>Comparative Analysis Results</h3>", unsafe_allow_html=True)
    
    def get_risk_markup(pred_val):
        if pred_val >= 0.50:
            return '<div class="risk-badge risk-high">⚠️ High Risk</div>'
        else:
            return '<div class="risk-badge risk-low">✅ Low Risk</div>'
            
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
        <div class="predict-box">
            <div class="model-title">📈 Linear Regression</div>
            <div class="prediction-value">{lr_pred:.4f}</div>
            {get_risk_markup(lr_pred)}
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
        <div class="predict-box">
            <div class="model-title">🌳 Random Forest</div>
            <div class="prediction-value">{rf_pred:.4f}</div>
            {get_risk_markup(rf_pred)}
        </div>
        """, unsafe_allow_html=True)
