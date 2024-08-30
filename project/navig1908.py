import streamlit as st

pg=st.navigation([

st.Page("dia.py",title="Diabetes Data Dashboard"),
st.Page("mob.py",title="Mobile Data Analysis"),
st.Page("mobpred.py",title="Mobile Data Prediction"),
st.Page("heart.py",title="Heart Data Analysis"),
st.Page("almondmain.py",title="Almond Data Analysis"),
st.Page("almonddpred.py",title="Almond Data Prediction"),





])

pg.run()