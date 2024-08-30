import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')


st.set_page_config(page_title="Diabetes Data Dashboard", page_icon=":bar_chart:", layout="wide")


st.title(":bar_chart: Diabetes Data Dashboard")


df = pd.read_csv('project/diabetes.csv')


st.subheader("First 5 Rows of Diabetes Data")
st.dataframe(df.head())


st.sidebar.header("Choose Your Filter:")
Pregnancies = st.sidebar.slider("Select Number of Pregnancies", int(df['Pregnancies'].min()), int(df['Pregnancies'].max()), (int(df['Pregnancies'].min()), int(df['Pregnancies'].max())))
Outcome = st.sidebar.multiselect("Select Outcome", df['Outcome'].unique())
AgeRange = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))


df_filtered = df[(df['Pregnancies'] >= Pregnancies[0]) & (df['Pregnancies'] <= Pregnancies[1]) & (df['Age'] >= AgeRange[0]) & (df['Age'] <= AgeRange[1])]
if Outcome:
    df_filtered = df_filtered[df_filtered['Outcome'].isin(Outcome)]


col1, col2, col3 = st.columns(3)

col1.subheader("Total Records")
col1.write(len(df_filtered))

col2.subheader("Average Age")
col2.write(round(df_filtered['Age'].mean(), 2))

col3.subheader("Average Glucose Level")
col3.write(round(df_filtered['Glucose'].mean(), 2))


st.subheader("Glucose Level Distribution")
fig1 = px.histogram(df_filtered, x="Glucose", title="Distribution of Glucose Levels", nbins=20, color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig1, use_container_width=True)


st.subheader("Distribution of Outcomes")
outcome_count = df_filtered['Outcome'].value_counts().reset_index()
outcome_count.columns = ['Outcome', 'Count']
fig2 = px.pie(outcome_count, values='Count', names='Outcome', title="Outcome Distribution", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig2, use_container_width=True)


st.subheader("Age vs Glucose Level")
fig3 = px.scatter(df_filtered, x="Age", y="Glucose", color="Outcome", title="Age vs Glucose Level", color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig3, use_container_width=True)


st.subheader("BMI Distribution")
fig4 = px.histogram(df_filtered, x="BMI", title="Distribution of BMI", nbins=20, color_discrete_sequence=['#EF553B'])
st.plotly_chart(fig4, use_container_width=True)


st.subheader("Top 10 Glucose Levels")
top_glucose = df_filtered.nlargest(10, 'Glucose')
fig5 = px.bar(top_glucose, x='Glucose', y='Age', color='Outcome', title="Top 10 Glucose Levels by Age", color_discrete_sequence=px.colors.qualitative.Dark2)
st.plotly_chart(fig5, use_container_width=True)


st.subheader("Filtered Data Table")
st.dataframe(df_filtered)
