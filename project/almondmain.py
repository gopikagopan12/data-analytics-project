import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Streamlit page configuration
st.set_page_config(page_title="Almond Dataset Analysis", page_icon="🌰", layout="wide")
st.title("🌰 Almond Data Analysis")

# Loading the almond dataset
adf = pd.read_csv('project/Almond.csv')
st.header("Almond Dataset")
st.dataframe(adf.head())

# Setting numerical labels (if needed)
st.subheader("Converting Data to Numerical Labels (if needed)")
le = LabelEncoder()
for column in adf.columns:
    if adf[column].dtype == 'object':
        adf[column] = le.fit_transform(adf[column])
st.dataframe(adf.head())

# Displaying summary statistics
st.subheader("Summary Statistics")
st.write(adf.describe())

# Pairplot of the almond dataset
st.subheader("Pairplot")
pairplot = sns.pairplot(adf)  # Adjust the hue parameter if there is a target column
st.pyplot(pairplot)

# Correlation heatmap
plt.figure(figsize=(10, 6))
st.subheader("Correlation Heatmap")
heatmap = sns.heatmap(adf.corr(), annot=True, cmap='coolwarm')
st.pyplot(heatmap.figure)
