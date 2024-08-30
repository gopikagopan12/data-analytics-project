import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as mat
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split as tts

st.set_page_config(page_title="Almond Classification", page_icon="🌰", layout="wide")

st.title("🌰 Almond Data Analysis and Classification")


almond_df = pd.read_csv('project/Almond.csv')
st.header("Almond Dataset")
st.dataframe(almond_df.head())


st.subheader("Column Names")
st.write(almond_df.columns)


le = LabelEncoder()
for column in almond_df.columns:
    if almond_df[column].dtype == 'object':  
        almond_df[column] = le.fit_transform(almond_df[column])

st.subheader("Encoded Dataset")
st.dataframe(almond_df.head())


x = almond_df.drop('Type', axis=1) 
y = almond_df['Type']  

almond_model = dtc(criterion='entropy', random_state=0)


xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2, random_state=42)
almond_model.fit(xtrain, ytrain)


ypred = almond_model.predict(xtest)


st.header("Classification Report")
st.table(mat.classification_report(ytest, ypred, output_dict=True))


st.header("Prediction Input")
input_data = []
for column in x.columns:
    min_val = float(x[column].min())
    max_val = float(x[column].max())
    input_val = st.number_input(f"Enter {column}", min_value=min_val, max_value=max_val)
    input_data.append(input_val)

sample1 = [input_data]

if st.button("Predict Almond Classification"):
    target_sp = almond_model.predict(sample1)
    proba = almond_model.predict_proba(sample1)
    st.write("Probability:", proba)
    st.write("Prediction (replace with actual target names):", target_sp)
    if target_sp == 1:
        st.write("This sample is likely to belong to the positive class.")
    else:
        st.write("This sample is unlikely to belong to the positive class.")

