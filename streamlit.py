import streamlit as st
import numpy as np
import pickle
import scikit-learn
import xgboost

# Load models
try:
    kidney_model = pickle.load(open('random.pkl', 'rb'))
    liver_model = pickle.load(open('liverxg.pkl', 'rb'))
    parkinson_model = pickle.load(open('parkinxg.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Display title and image
st.image("C:/Users/MR B/Downloads/streamlt image.png", width=90)
st.markdown(
    "<h1 style='text-align: center;'>Guvi Data Science - Multiple Diseases Prediction Dashboard</h1>",
    unsafe_allow_html=True,
)

# Sidebar for selecting disease model
st.sidebar.title("Choose a Disease Model")
menu = st.sidebar.selectbox(
    "Select a Disease:", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"]
)

# Kidney Disease Prediction
if menu == "Kidney Disease":
    st.title("Kidney Disease Prediction")
    st.header("Enter Patient Details")

    # Input fields
    age = st.number_input("Age:", min_value=0, max_value=100, step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg):", min_value=0, max_value=200, step=1)
    specific_gravity = st.selectbox("Specific Gravity:", [1.005, 1.010, 1.015, 1.020, 1.025])
    albumin = st.slider("Albumin Level:", min_value=0, max_value=5, step=1)
    sugar = st.slider("Sugar Level:", min_value=0, max_value=5, step=1)
    red_blood_cells = st.selectbox("Red Blood Cells:", ["normal", "abnormal"])
    pus_cells = st.selectbox("Pus Cells:", ["normal", "abnormal"])
    pus_cell_clumps = st.selectbox("Pus Cell Clumps:", ["present", "notpresent"])
    bacteria = st.selectbox("Bacteria:", ["present", "notpresent"])
    blood_glucose_random = st.number_input("Blood Glucose Random (mg/dL):", min_value=0, max_value=500, step=1)
    packed_cell_volume = st.number_input("Packed Cell Volume:", min_value=0, max_value=100, step=1)
    white_blood_cell_count = st.number_input("White Blood Cell Count (per cubic mm):", min_value=0, max_value=20000, step=100)
    red_blood_cell_count = st.number_input("Red Blood Cell Count (millions per cubic mm):", min_value=0.0, max_value=10.0, step=0.1)
    hypertension = st.selectbox("Hypertension:", ["yes", "no"])
    diabetes_mellitus = st.selectbox("Diabetes Mellitus:", ["yes", "no"])
    coronary_artery_disease = st.selectbox("Coronary Artery Disease:", ["yes", "no"])
    appetite = st.selectbox("Appetite:", ["good", "poor"])
    pedal_edema = st.selectbox("Pedal Edema:", ["yes", "no"])
    anemia = st.selectbox("Anemia:", ["yes", "no"])

    # Encoding categorical variables
    mapping = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0, "yes": 1, "no": 0, "good": 1, "poor": 0}
    inputs = np.array([
        age, blood_pressure, specific_gravity, albumin, sugar,
        mapping[red_blood_cells], mapping[pus_cells], mapping[pus_cell_clumps],
        mapping[bacteria], blood_glucose_random, packed_cell_volume,
        white_blood_cell_count, red_blood_cell_count, mapping[hypertension],
        mapping[diabetes_mellitus], mapping[coronary_artery_disease], mapping[appetite],
        mapping[pedal_edema], mapping[anemia]
    ]).reshape(1, -1)

    # Predict button
    if st.button("Predict Kidney Disease"):
        prediction = random.predict(inputs)
        if prediction[0] == 1:
            st.error("The patient is likely to have kidney disease.")
        else:
            st.success("The patient is unlikely to have kidney disease.")

# Liver Disease Prediction
elif menu == "Liver Disease":
    st.title("Liver Disease Prediction")
    st.header("Enter Patient Details")

    # Input fields
    age = st.number_input("Age:", min_value=0, max_value=100, step=1)
    total_bilirubin = st.number_input("Total Bilirubin:", min_value=0.0, max_value=10.0, step=0.1)
    direct_bilirubin = st.number_input("Direct Bilirubin:", min_value=0.0, max_value=5.0, step=0.1)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase (U/L):", min_value=0, max_value=1000, step=1)
    alanine_aminotransferase = st.number_input("Alanine Aminotransferase (U/L):", min_value=0, max_value=500, step=1)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (U/L):", min_value=0, max_value=500, step=1)
    total_proteins = st.number_input("Total Proteins (g/dL):", min_value=0.0, max_value=10.0, step=0.1)
    albumin = st.number_input("Albumin (g/dL):", min_value=0.0, max_value=5.0, step=0.1)
    albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio:", min_value=0.0, max_value=5.0, step=0.1)

    # Predict button
    liver_inputs = np.array([
        age, total_bilirubin, direct_bilirubin, alkaline_phosphatase,
        alanine_aminotransferase, aspartate_aminotransferase, total_proteins,
        albumin, albumin_globulin_ratio
    ]).reshape(1, -1)
    if st.button("Predict Liver Disease"):
        prediction = liver_model.predict(liver_inputs)
        if prediction[0] == 1:
            st.error("The patient is likely to have liver disease.")
        else:
            st.success("The patient is unlikely to have liver disease.")

# Parkinson's Disease Prediction
elif menu == "Parkinson's Disease":
    st.title("Parkinson's Disease Prediction")
    st.header("Enter Patient Details")

    # Input fields
    MDVP_Fo_Hz = st.number_input("Fundamental Frequency (MDVP:Fo(Hz))", min_value=0.0, value=0.0)
    MDVP_Fhi_Hz = st.number_input("Maximum Frequency (MDVP:Fhi(Hz))", min_value=0.0, value=0.0)
    MDVP_Jitter_percent = st.number_input("Jitter (MDVP:Jitter(%))", min_value=0.0, value=0.0)
    MDVP_Shimmer = st.number_input("Shimmer (MDVP:Shimmer)", min_value=0.0, value=0.0)
    NHR = st.number_input("Noise-to-Harmonics Ratio (NHR)", min_value=0.0, value=0.0)

    # Predict button
    parkinson_inputs = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Jitter_percent, MDVP_Shimmer, NHR]])
    if st.button("Predict Parkinson's Disease"):
        prediction = parkinsons_model.predict(parkinson_inputs) # type: ignore
        if prediction[0] == 1:
            st.error("The patient is likely to have Parkinson's disease.")
        else:
            st.success("The patient is unlikely to have Parkinson's disease.")
