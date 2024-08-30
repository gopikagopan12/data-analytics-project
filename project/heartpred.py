import streamlit as st
import joblib


model = joblib.load('kmeans_heart.pkl')

st.header("Heart Disease Prediction")


col1, col2 = st.columns(2)

age = col1.number_input("Enter Age", min_value=1, max_value=120, value=50)
sex = col2.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = col1.selectbox("Chest Pain Type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", options=[0, 1, 2, 3])
trestbps = col2.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = col1.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
fbs = col2.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", options=[0, 1])
restecg = col1.selectbox("Resting ECG results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)", options=[0, 1, 2])
thalach = col2.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = col1.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", options=[0, 1])
oldpeak = col2.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0)
slope = col1.selectbox("Slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)", options=[0, 1, 2])
ca = col2.selectbox("Number of major vessels (0-3) colored by fluoroscopy", options=[0, 1, 2, 3])
thal = col1.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)", options=[1, 2, 3])


input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("The model predicts that this patient **has** heart disease.")
    else:
        st.success("The model predicts that this patient **does not have** heart disease.")
