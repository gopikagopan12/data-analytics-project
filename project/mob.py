import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn import metrics

st.set_page_config(page_title="Mobile Price Analysis", page_icon="📱", layout="wide")
st.title("📱 Mobile Price Data Analysis 📊")


sdf = pd.read_csv("project/Mobile-Price-Prediction-cleaned_data.csv")
st.subheader("Mobile Dataset")
st.dataframe(sdf.head())

st.header("Null values in data")
st.table(sdf.isnull().sum())

st.header("Statistical summary of data")
st.table(sdf.describe())

st.header("Columns of data")
st.write(list(sdf.columns))


training_data = sdf[sdf['Price'].isnull() == False]
testing_data = sdf[sdf['Price'].isnull() == False]


c1, c2 = st.columns(2)

c1.subheader("Shape of training data")
c1.write(training_data.shape)
c1.subheader("Null values in training data")
c1.write(training_data.isnull().sum())

c2.subheader("Shape of testing data")
c2.write(testing_data.shape)
c2.subheader("Null values in testing data")
c2.write(testing_data.isnull().sum())

c1.subheader("Training data")
c1.table(training_data.head())

c2.subheader("Testing data")
c2.table(testing_data.head())


xtrain = training_data.drop('Price', axis=1)
ytrain = training_data[['Price']]

xtest = testing_data.drop('Price', axis=1)
ytest = testing_data[['Price']]

c3, c4, c5, c6 = st.columns(4)

c3.subheader("Features of training data")
c3.table(xtrain.head())

c4.subheader("Labels of training data")
c4.table(ytrain.head())

c5.subheader("Features of testing data")
c5.table(xtest.head())

c6.subheader("Labels of testing data")
c6.table(ytest.head())


rid = Ridge()
lass = Lasso()
enet = ElasticNet()


rid.fit(xtrain, ytrain)
lass.fit(xtrain, ytrain)
enet.fit(xtrain, ytrain)


pickle.dump(rid, open('rid11.pkl', 'wb'))
pickle.dump(lass, open('lass11.pkl', 'wb'))
pickle.dump(enet, open('enet11.pkl', 'wb'))


ypred1 = rid.predict(xtest)
ypred2 = lass.predict(xtest)
ypred3 = enet.predict(xtest)

st.header("Comparison of different models")

st.subheader("R2 score")
r21 = metrics.r2_score(ytest, ypred1)
r22 = metrics.r2_score(ytest, ypred2)
r23 = metrics.r2_score(ytest, ypred3)

col1, col2, col3 = st.columns(3)
col1.write(r21)
col2.write(r22)
col3.write(r23)

st.subheader("MSE ")
mse1 = metrics.mean_squared_error(ytest, ypred1)
mse2 = metrics.mean_squared_error(ytest, ypred2)
mse3 = metrics.mean_squared_error(ytest, ypred3)

col1, col2, col3 = st.columns(3)
col1.write(mse1)
col2.write(mse2)
col3.write(mse3)

st.subheader("MAE ")
mae1 = metrics.mean_absolute_error(ytest, ypred1)
mae2 = metrics.mean_absolute_error(ytest, ypred2)
mae3 = metrics.mean_absolute_error(ytest, ypred3)

col1, col2, col3 = st.columns(3)
col1.write(mae1)
col2.write(mae2)
col3.write(mae3)

st.header("Prediction of different models")
testing_data['Ridge_Price'] = ypred1
testing_data['Lasso_Price'] = ypred2
testing_data['Enet_Price'] = ypred3

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ypred1, c='g', marker='+', label='Ridge')
ax.plot(ypred2, c='b', marker='*', label='Lasso')
ax.plot(ypred3, c='r', marker='o', label='ElasticNet')
ax.legend()

st.pyplot(fig)
