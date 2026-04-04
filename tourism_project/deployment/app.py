import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="SachinAbi/tourism_package_prediction_model",
    filename="tourism_package_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**.

Enter customer details and interaction information below to get a prediction.
""")

# ---------------------------
# Customer Details
# ---------------------------
st.header("Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
passport = st.selectbox("Has Passport", [0, 1])
own_car = st.selectbox("Owns Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=0, value=30000)

# ---------------------------
# Interaction Details
# ---------------------------
st.header("Customer Interaction Details")

pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=30)

# ---------------------------
# Prepare Input Data
# ---------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch
}])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")

    if prediction == 1:
        st.success("✅ Customer is likely to PURCHASE the Wellness Tourism Package")
    else:
        st.warning("❌ Customer is NOT likely to purchase the package")
