
import streamlit as st
import pandas as pd
import pickle

# Load Model and Scaler
model = pickle.load(open("best_diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page Title
st.title("🏥 Diabetes Prediction System")

st.write("Enter patient details below:")

# Input Fields
gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.selectbox(
    "Smoking History",
    [
        "No Info",
        "current",
        "ever",
        "former",
        "never",
        "not current"
    ]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=100.0,
    value=25.0
)

HbA1c_level = st.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.5
)

blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=100
)

# Encode Inputs
gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_map = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}

gender = gender_map[gender]
smoking_history = smoking_map[smoking_history]

# Prediction Button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [HbA1c_level],
        "blood_glucose_level": [blood_glucose_level]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

# Insights Section
st.markdown("---")

st.subheader("Key Risk Factors")

st.write("""
1. High HbA1c Level
2. High Blood Glucose Level
3. High BMI
4. Increasing Age
5. Hypertension
6. Heart Disease
""")


