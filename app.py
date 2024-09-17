import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


#loading the model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('onehot_encoder.pkl', 'rb') as file:
    onehot_encoder=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

##StreamLit App

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 15, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Number')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credict Card', [0, 1])
is_active_member = st.selectbox('Is acitve Member', [0, 1])

#prepare inpiut data

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age ],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': has_cr_card,
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

pred = model.predict(input_data_scaled)
pred_probability = pred[0][0]

st.write(f'Churn Probability: {pred_probability:.3f}')

if pred_probability > 0.5:
    st.write('The customer is likely to churn 👎 ')
else:
    st.write('The customer is not likely to churn 👍 ')

