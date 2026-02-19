import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

import tensorflow as tf
import pickle

## Load the Model
model = tf.keras.models.load_model('model.h5')

## load the Encoders
with open('encoder/label_encode_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('encoder/ohe_geo.pkl','rb') as file:
    geo_encoder = pickle.load(file)

##Load the Scalar pickle
with open('Scaler/scaler.pkl','rb') as file:
    scaler = pickle.load(file)


## Streamlit App
st.title("Customer Churn Prediction")

#user Inputs
creditScore = st.number_input('CreditScore')
geography = st.selectbox('Geography',geo_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')
numOfProducts= st.slider('Num Of Products',1,4)
hasCrCard = st.selectbox('Has Credit Card',[0,1])
isActiveMember = st.selectbox('Is active member',[0,1])
estimatedSalary = st.number_input('EstimatedSalary')

## Input Data
input_data = pd.DataFrame({
    "CreditScore" : [creditScore],
    "Gender" : [label_encoder_gender.transform([gender])[0]],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts": [numOfProducts],
    "HasCrCard" : [hasCrCard],
    "IsActiveMember" : [isActiveMember],
    "EstimatedSalary" : [estimatedSalary]
})

## Encoding The Geography and Appending to main input data frame

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out())


##combining
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Saclling the whole data
input_data_scaled = scaler.transform(input_data)

## Prediction churn
pred = model.predict(input_data_scaled)
prediction_proba = pred[0][0]

st.write(f"Churn Probability : {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The Customer is likely to Chrun")
else:
    st.write("The Customer is not likely to churn the bank")

