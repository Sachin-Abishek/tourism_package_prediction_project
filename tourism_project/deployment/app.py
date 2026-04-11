import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="SachinAbi/tourism_package_prediction_model",
    filename="tourism_package_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**.
""")

# --- Input UI ---
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips per Year", min_value=0.0, max_value=20.0, value=2.0)
    passport = st.selectbox("Has Passport (0=No, 1=Yes)", [0, 1])
    own_car = st.selectbox("Owns Car (0=No, 1=Yes)", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=0.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)

st.header("Interaction Details")
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=2.0)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=120.0, value=15.0)

# Ensure DataFrame columns match training X exactly
input_dict = {
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}

input_data = pd.DataFrame([input_dict])

if st.button("Predict Purchase"):
    try:
        classification_threshold = 0.45
        prediction_proba = model.predict_proba(input_data)[:, 1]
        prediction = (prediction_proba >= classification_threshold).astype(int)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"✅ Purchase Likely (Probability: {prediction_proba[0]:.2f})")
        else:
            st.warning(f"❌ Purchase Unlikely (Probability: {prediction_proba[0]:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
