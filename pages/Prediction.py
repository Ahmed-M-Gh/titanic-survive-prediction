#importing the file
import streamlit as st
import joblib as jb
import pandas as pd

#identify the models
model = jb.load(r'result/model_train_result.sav')
LE = jb.load(r'result/Label_encoding_sex.sav')
OE = jb.load(r'result/One_Hot_encoding_embarked.sav')
scalar = jb.load(r'result/Standared_Scalar.sav')
column_after_encoding = jb.load(r'result/column_encoded_result.sav')

#adding a title and subtitle
st.set_page_config(page_title='Prediction', page_icon='🔍')
st.title("🔍 Titanic Survival Prediction")
st.info("This program predicts if a person survived or not based on input data.")

#identify our inputs
Pclass = st.number_input('Pclass', min_value=1, max_value=3, step=1, format='%d')
Sex = st.selectbox('Gender', ['male', 'female'])
Age = st.slider('Age', min_value=1, max_value=100)
SibSp = st.number_input('SibSp', step=1)
Parch = st.number_input('Parch', step=1)
Fare = st.number_input('Fare', min_value=0.0, max_value=500.0, step=0.1, format='%.2f')
Embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

#convert them to a dataframe
df = pd.DataFrame({
    'Pclass' : [Pclass],
    'Sex' : [Sex],
    'Age' : [Age],
    'SibSp' : [SibSp],
    'Parch' : [Parch],
    'Fare' : [Fare],
    'Embarked' : [Embarked]
})
#making a button for start operation and prediction
con = st.button('Start predicting')
if con:
    #label encoding
    df['Sex'] = LE.transform(df['Sex'])
    #one-hot encoding
    embarked_encoded = pd.DataFrame(OE.transform(df[['Embarked']]),
                                    columns = OE.get_feature_names_out(['Embarked']))
    #concatenate data and sort it as the train time
    df = pd.concat([df.drop('Embarked', axis=1), embarked_encoded], axis=1)
    df = df.reindex(columns=column_after_encoding)
    #Standariz our data
    df_scaled = scalar.transform(df)
    #get the predict
    prediction = model.predict(df_scaled)[0]
    probability = round(float(model.predict_proba(df_scaled)[0][prediction]), 3)
    #making the output in a nice way
    tap1, tap2 = st.tabs(['Prediction 💡', 'Details 📊'])
    with tap1:
        if prediction == 1:
            st.success('The person survived ✅')
        else :
            st.error('The preson did not survive  💀')
    with tap2:
        st.info(f'The Prediction Result is : {prediction}')
        st.success(f'📌 The probability of survival is : {probability}')
        st.error(f'📌 The probability of not survival is : {probability}')
        


