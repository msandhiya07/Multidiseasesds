import streamlit as st
import numpy as np
import pickle 
import pandas as pd


st.image("C:/Users/MR B/Downloads/streamlt image.png",width=90)#type: ignore
st.markdown("<h1 style='text-align: center;'> Guvi Data Science</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> Multiple diseases Dashboard </h2>", unsafe_allow_html=True)


with open('kidney_model.pkl', 'rb') as f:
    kidney_model = pickle.load(f)
with open('liver_model.pkl', 'rb') as f:
    liver_model = pickle.load(f)
with open('parkinsons_model.pkl', 'rb') as f:
    parkinsons_model = pickle.load(f)

# Sidebar Navigation
menu = st.sidebar.radio("Choose Disease to Predict:", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])

# Kidney Disease Prediction
if menu == "Kidney Disease":
    st.header('Predict Kidney Disease')
sg = st.number_input('Specific Gravity', min_value=1.0, max_value=1.05, step=0.01)
al = st.number_input('Albumin Level', min_value=0, max_value=5, step=1)
sc = st.number_input('Serum Creatinine (mg/dL)', min_value=0.0, max_value=15.0, step=0.1)
hemo = st.number_input('Hemoglobin', min_value=5.0, max_value=20.0, step=0.1)
pcv = st.number_input('Packed Cell Volume', min_value=10, max_value=50, step=1)
sodium = st.number_input('Sodium Level', min_value=100.0, max_value=200.0, step=0.1)
potassium = st.number_input('Potassium Level', min_value=2.0, max_value=10.0, step=0.1)
rbc = st.number_input('Red Blood Cell Count', min_value=3.0, max_value=6.0, step=0.1)
wc = st.number_input('White Blood Cell Count', min_value=4000, max_value=12000, step=100)
bp = st.number_input('Blood Pressure', min_value=80, max_value=200, step=1)

inputs = np.array([[sg, al, sc, hemo, pcv, sodium, potassium, rbc, wc, bp]])
inputs = np.array([[sg, al, sc, hemo, pcv,sodium, potassium, rbc, wc, bp]])
if st.button('Predict'):
        result = kidney_model.predict(inputs)
        st.write("Prediction:", 'Positive' if result[0] == 1 else 'Negative')

# Liver Disease Prediction

elif menu == "Liver Disease":
    st.header('Predict Liver Disease')
    print(parkinsons_model.n_features_in_)
    age = st.number_input('Age', min_value=10, max_value=100, step=1)
    tb = st.number_input('Total Bilirubin', min_value=0.1, max_value=10.0, step=0.1)
    db = st.number_input('Direct Bilirubin', min_value=0.0, max_value=5.0, step=0.1)
    alkphos = st.number_input('Alkaline Phosphotase', min_value=10, max_value=400, step=1)
    sgpt = st.number_input('SGPT', min_value=0, max_value=300, step=1)
    sgot = st.number_input('SGOT', min_value=0, max_value=300, step=1)
    inputs = np.array([[age, tb, db, alkphos, sgpt, sgot]])
    if st.button('Predict'):
        result = liver_model.predict(inputs)
        st.write("Prediction:", 'Positive' if result[0] == 1 else 'Negative')

# Parkinson's Disease Prediction
elif menu == "Parkinson's Disease":
    st.header("Predict Parkinson's Disease")
fo = st.number_input('MDVP:Fo(Hz)', min_value=50.0, max_value=300.0, step=1.0)
fhi = st.number_input('MDVP:Fhi(Hz)', min_value=60.0, max_value=500.0, step=1.0)
flo = st.number_input('MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, step=1.0)
jitter = st.number_input('Jitter(%)', min_value=0.0, max_value=1.0, step=0.01)
shimmer = st.number_input('Shimmer', min_value=0.0, max_value=1.0, step=0.01)
apq = st.number_input('APQ', min_value=0.0, max_value=1.0, step=0.01)
dda = st.number_input('DDA', min_value=0.0, max_value=1.0, step=0.01)
spread1 = st.number_input('Spread1', min_value=-8.0, max_value=0.0, step=0.1)
spread2 = st.number_input('Spread2', min_value=0.0, max_value=7.0, step=0.1)
ppe = st.number_input('PPE', min_value=0.0, max_value=0.7, step=0.01)
inputs = np.array([[fo, fhi, flo, jitter, shimmer,apq,dda,spread1,spread2,ppe]])
if st.button('Predict'):
        result = parkinsons_model.predict(inputs)
        st.write("Prediction:", 'Positive' if result[0] == 1 else 'Negative')
