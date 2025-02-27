import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
#import plotly .express as px



st.title('BANK ACCOUNT')
st.subheader('Built by Ifeyinwa')

Bank = pd.read_csv('Financial_inclusion_dataset.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>ADVERT SALES PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by IFEYINWA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com.png')
st.divider()

st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("Financial institutions often need to predict whether a customer will open a new bank account based on their demographic, financial, and behavioral data. Accurate prediction models can help banks tailor their marketing strategies, improve customer outreach, and optimize product offerings. The goal is to develop a predictive model that identifies the likelihood of a customer opening a bank account, leveraging historical data such as age, income, employment status, transaction history, and customer interactions.")

st.divider()

st.dataframe(Bank,use_container_width= True)

st.sidebar.image('bank user icon.png',caption = "Welcome User")


#['age_of_respondent','household_size','job_type','education_level','country','relationship_with_head','gender_of_respondent','bank_account']

age= st.sidebar.number_input('age_of respondent exp', min_value=0.0, max_value=100.0, value=Bank.age_of_respondent.median())
house = st.sidebar.number_input('Household exp', min_value= 1.0, max_value = 10.0, value=Bank.household_size.median())
job = st.sidebar.selectbox('Type of Job', Bank.job_type.unique(),index=1)
education= st.sidebar.selectbox('Eduacation level', Bank.education_level.unique(), index =1)
cntry = st.sidebar.selectbox('country', Bank.country.unique(),index=1)
relation = st.sidebar.selectbox('relationship_with_head', Bank.relationship_with_head.unique(),index=1)
gender= st.sidebar.selectbox('Gender_of_respondent',Bank.gender_of_respondent.unique(),index=1)



 #user input,we want to recognise the original name given in the dataset and link it

inputs = {

    'age_of_respondent' : [age],    
    'household_size' : [house],
    'job_type' : [job],
    'education_level' : [education],    
    'country' : [cntry],
    'relationship_with_head' : [relation],
     'gender_of_respondent': [gender]
    
}


 #if we want the input  to appear under the  dataset

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)

# transform the user inputs,import the transformers(scalers)

age_scaler = joblib.load('age_of_respondent_scaler.pkl')
house_scaler = joblib.load('household_size_scaler.pkl')
job_encoder = joblib.load('job_type_encoder.pkl')
education_encoder = joblib.load('education_level_encoder.pkl')
cntry_encoder = joblib.load('country_encoder.pkl')
relation_encoder = joblib.load('relationship_with_head_encoder.pkl')
gender_encoder = joblib.load('gender_of_respondent_encoder.pkl')


inputVar['age_of_respondent'] = age_scaler.transform(inputVar[['age_of_respondent']])
inputVar['household_size'] = house_scaler.transform(inputVar[['household_size']])
inputVar['job_type'] = job_encoder.transform(inputVar[['job_type']])
inputVar['education_level'] = education_encoder.transform(inputVar[['education_level']])
inputVar['country'] = cntry_encoder.transform(inputVar[['country']])
inputVar['relationship_with_head'] = relation_encoder.transform(inputVar[['relationship_with_head']])
inputVar['gender_of_respondent'] = gender_encoder.transform(inputVar[['gender_of_respondent']])









 ##Bringing in the model
model = joblib.load('Bankmodel.pkl')

# we create a button to use for the prediction


predictbutton = st.button('Push to Predict if the customer has account number')
 
if predictbutton: 
    predicted = model.predict(inputVar)
    st.success(f'the Customer has a  : {predicted} account number')




