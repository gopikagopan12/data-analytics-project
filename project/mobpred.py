import streamlit as st
import pickle

st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱", layout="wide")
st.title("📱 Mobile Price Prediction")


model_ridge = pickle.load(open('rid11.pkl', 'rb'))
model_lasso = pickle.load(open('lass11.pkl', 'rb'))
model_elasticnet = pickle.load(open('enet11.pkl', 'rb'))

st.header("Prediction")


n1 = float(st.number_input("Enter value for Ratings: "))
n2 = int(st.number_input("Enter value for RAM (GB): "))
n3 = int(st.number_input("Enter value for ROM (GB): "))
n4 = float(st.number_input("Enter value for Mobile Size (inches): "))
n5 = int(st.number_input("Enter value for Primary Camera (MP): "))
n6 = int(st.number_input("Enter value for Selfie Camera (MP): "))
n7 = int(st.number_input("Enter value for Battery Power (mAh): "))


sample1 = [[n1, n2, n3, n4, n5, n6, n7]]

if st.button("Predict the price"):
    
    t1 = model_ridge.predict(sample1)
    t2 = model_lasso.predict(sample1)
    t3 = model_elasticnet.predict(sample1)
    
    
    if t1 is not None:
        st.write("Predicted price of the mobile is:")
        c1, c2, c3 = st.columns(3)
        c1.subheader("Ridge Regression")
        c2.subheader("LASSO Regression")
        c3.subheader("ElasticNet Regression")
        c1.write(t1[0])
        c2.write(t2[0])
        c3.write(t3[0])
    else:
        st.write("Price cannot be determined")