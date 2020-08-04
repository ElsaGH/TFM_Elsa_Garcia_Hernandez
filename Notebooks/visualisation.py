import numpy as np
from statistics import median
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import lightgbm as lgb
from lightgbm import LGBMClassifier


X_train = pd.read_pickle('./Data/X_train.pickle')
X_test = pd.read_pickle('./Data/X_test.pickle')
y_train = pd.read_pickle('./Data/y_train.pickle')
y_test = pd.read_pickle('./Data/y_test.pickle')



# Train the model        
clfLGB = LGBMClassifier(class_weight='balanced', random_state=0, objective='binary', n_estimators=800, learning_rate=0.03,
        max_depth=4, min_data_in_leaf=81, num_leaves=9, min_child_weight=0.001, feature_fraction=0.9)
clfLGB.fit(X_train, y_train)
# y_predicted = clfLGB.predict(X_test)
# y_predicted_proba_clfLGB = clfLGB.predict_proba(X_test)
    

# Output
def main():
    st.title('Colon cancer survival')
    st.header('Predicting the probability of survival at two years after colon cancer diagnosis')
    st.write('This tool is designed to help to predict the probability of survival at two years after colon cancer diagnosis. First, this tool will ask you for some details about the patient and the tumour. Then, it will use a Light Gradient Boosting Machine model to show the probability of survival, based on data of similar patients in the past.')
    st.write('*************************************************************************************')
    st.subheader('Please enter the data of the patient:')
    sex = st.radio('What is the gender of the patient?',('Male','Female'))
    if sex == 'Male':
        sex = 1
    else:
        sex = 0
    number_tumours = st.selectbox('How many tumours does the patient have in the colon?', ['One tumour', 'Two tumours', 'Three tumours'])
    
    if number_tumours == 'One tumour':
        n_tumours = 1
    elif number_tumours == 'Two tumours':
        n_tumours  = 2
    elif number_tumours == 'Three tumours':
        n_tumours = 3
    
    
# One tumour:
    
    if n_tumours == 1:
        bmi = st.slider('What was the BMI of the patient at the time of diagnosis?', 0, 80, 25)
        age = st.slider('What was the age of the patient at the time of diagnosis?', 0, 110, 50)
        surgery_yn = st.radio('Did the patient undergo any surgery procedure to remove the tumour?',('Yes','No'))
        if surgery_yn == 'Yes':
            diag_to_surg = st.slider('How many days elapsed between the tumour diagnosis and the surgery?', 0, 300, 30)
        else:
            diag_to_surg = X_train['DIAG_TO_SURG_DAYS_MEDIAN'].median()
        stage = st.selectbox('What is the stage of the tumour?', ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
        grade = st.selectbox('What is the grade of the tumour?', ['Grade 1 - Well differentiated', 'Grade 2 - Moderately differentiated', 'Grade 3 - Poorly differentiated', 'Grade 4 - Undifferentiated'])

        stage_final = 0
        if stage == 'Stage 0':
            stage_final = 0
        elif stage == 'Stage 1':
            stage_final = 1
        elif stage == 'Stage 2':
            stage_final = 2
        elif stage == 'Stage 3':
            stage_final = 3
        elif stage == 'Stage 4':
            stage_final = 4
        else:
            print('Error')


        grade_final = 0
        if grade == 'Grade 1 - Well differentiated':
            grade_final = 1
        elif grade == 'Grade 2 - Moderately differentiated':
            grade_final = 2
        elif grade == 'Grade 3 - Poorly differentiated':
            grade_final = 3
        elif grade == 'Grade 4 - Undifferentiated':
            grade_final = 4
        else:
            print('Error')

        user_data = np.array([n_tumours, bmi, age, diag_to_surg, sex, stage_final, grade_final]).reshape(1,-1)
        y_predicted_proba_clfLGB = clfLGB.predict_proba(user_data)

        if st.button('Submit'):
            outcome = str(round(y_predicted_proba_clfLGB[0,0] * 100,1))
            st.write('*************************************************************************************')
            st.write('At two years after diagnosis, the patient has a probability of survival of ',outcome,'%')

            outcome = float(outcome)
            if outcome <= 25:
                color = 'red'
            elif 25 < outcome <= 50:            
                color = 'orange'
            elif 50 < outcome <= 75:            
                color = 'yellow'
            elif outcome > 75:            
                color = 'green'


            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                number = {'suffix': "%"},
                value = outcome,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': 'Probability of survival'},
                gauge = {'axis': {'range': [0, 100]},
                        'bar': {'color': color}}))
            st.write(fig)

            
# More than one tumour:            
            
    else:
        bmi = st.slider('What was the average BMI of the patient, considering the different BMI values that the patient had when each tumour was diagnosed?', 0, 80, 25)
        age = st.slider('What was the average age of the patient, considering the different ages of the patient when each tumour was diagnosed?', 0, 110, 50)
        surgery_yn = st.radio('Did the patient undergo any surgery procedure to remove any of the tumours?',('Yes','No'))
        if surgery_yn == 'Yes':
            diag_to_surg = st.slider('On average, how many days elapsed between the tumour diagnosis and the surgery, considering the different days elapsed for each of tumour?', 0, 300, 30)
        else:
            diag_to_surg = X_train['DIAG_TO_SURG_DAYS_MEDIAN'].median()
        stage = st.selectbox('What is the highest stage of the tumours?', ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
        grade = st.selectbox('What is the highest grade of the tumours?', ['Grade 1 - Well differentiated', 'Grade 2 - Moderately differentiated', 'Grade 3 - Poorly differentiated', 'Grade 4 - Undifferentiated'])

        stage_final = 0
        if stage == 'Stage 0':
            stage_final = 0
        elif stage == 'Stage 1':
            stage_final = 1
        elif stage == 'Stage 2':
            stage_final = 2
        elif stage == 'Stage 3':
            stage_final = 3
        elif stage == 'Stage 4':
            stage_final = 4
        else:
            print('Error')


        grade_final = 0
        if grade == 'Grade 1 - Well differentiated':
            grade_final = 1
        elif grade == 'Grade 2 - Moderately differentiated':
            grade_final = 2
        elif grade == 'Grade 3 - Poorly differentiated':
            grade_final = 3
        elif grade == 'Grade 4 - Undifferentiated':
            grade_final = 4
        else:
            print('Error')

        user_data = np.array([n_tumours, bmi, age, diag_to_surg, sex, stage_final, grade_final]).reshape(1,-1)
        y_predicted_proba_clfLGB = clfLGB.predict_proba(user_data)

        if st.button('Submit'):
            outcome = str(round(y_predicted_proba_clfLGB[0,0] * 100,1))
            st.write('*************************************************************************************')
            st.write('At two years after diagnosis, the patient has a probability of survival of ',outcome,'%')

            outcome = float(outcome)
            if outcome <= 25:
                color = 'red'
            elif 25 < outcome <= 50:            
                color = 'orange'
            elif 50 < outcome <= 75:            
                color = 'yellow'
            elif outcome > 75:            
                color = 'green'


            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                number = {'suffix': "%"},
                value = outcome,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': 'Probability of survival'},
                gauge = {'axis': {'range': [0, 100]},
                        'bar': {'color': color}}))
            st.write(fig)

            
main()