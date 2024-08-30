import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as mat
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Heart Disease Data Analysis", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Data Analysis ❤️")


df = pd.read_csv('heart.csv')


st.header('❤️ Heart Disease Data Set ❤️')
st.table(df.head())


cl1, cl2 = st.columns(2)

cl1.header("Count of unique target labels")
cl1.table(df['target'].value_counts())


x = df.drop(columns=['target']) 
y = df[['target']]  


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


wcss = []
k = []
for i in range(1, 11):
    k.append(i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

st.write("Values of k and WCSS")
fig, ax = plt.subplots(figsize=(2, 2))
ax.plot(k, wcss, c='g', marker='o', mfc='r')
st.pyplot(fig)


km_final = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0)
df['Cluster_Label'] = km_final.fit_predict(x_scaled)


st.header('❤️ Heart Disease Data Set with Cluster Labels ❤️')
st.table(df)


st.header("Visualizing the new labels and clusters")

fig2 = px.scatter(df, x='age', y='trestbps', size='chol', color='Cluster_Label', title="Clusters by Age and Resting Blood Pressure")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x='age', y='trestbps', size='chol', color='target', title="Original Labels by Age and Resting Blood Pressure")
st.plotly_chart(fig3, use_container_width=True)


dbs = mat.davies_bouldin_score(x_scaled, km_final.labels_)
sil = mat.silhouette_score(x_scaled, km_final.labels_)
cal = mat.calinski_harabasz_score(x_scaled, km_final.labels_)

ars = mat.adjusted_rand_score(df['target'], km_final.labels_)
mu = mat.mutual_info_score(df['target'], km_final.labels_)

st.header("Evaluation Scores")

c1, c2, c3 = st.columns(3)
c4, c5 = st.columns(2)

c1.subheader('Davies-Bouldin Score')
c1.subheader(dbs)
c2.subheader('Silhouette Score')
c2.subheader(sil)
c3.subheader('Calinski-Harabasz Score')
c3.subheader(cal)
c4.subheader('Adjusted Rand Score')
c4.subheader(ars)
c5.subheader('Mutual Info Score')
c5.subheader(mu)


pickle.dump(km_final, open('kmeans_heart.pkl', 'wb'))
