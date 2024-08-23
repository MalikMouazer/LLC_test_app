import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib


import sys
print("le programme s'éxécute dans : --->  ", sys.executable)

#############################
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
################"#############"


st.set_page_config(layout="wide")

# Charger le modèle sauvegardé
@st.cache_resource
def get_model(model_name):
    return joblib.load(model_name)



input_keys_1 = ['age', 'nb_anomalies_k', 'gain_2p', 'anomalie_myc', 'sexe', 'del_17p',
       'del_11q', 'tri_12', 'stade_diag', 'del_13q', 'del_8p', 'statut_IGHV']

input_keys = ['age', 'sexe', 'nb_anomalies_k', 'del_17p', 'del_11q', 'tri_12',
       'stade_diag_b_ou_c', 'del_13q', 'del_8p', 'gain_2p', 'statut_IGHV',
       'anomalie_myc']



# print('\n'.join(sidebar_code))
if 'patients' not in st.session_state:
    st.session_state['patients'] = []
if 'display' not in st.session_state:
    st.session_state['display'] = 1
if 'model' not in st.session_state:
    st.session_state['model'] = 'mas_model' # POur choisir le nom du modèle par défaut ? 
    
#mas_model = get_model('diag_dn_popg_cph_v0.pkl')
mas_model = get_model('cyto_dn_popg_RSF_v0.pkl')
# mas_model = get_model(st.session_state['model'])  : 


def plot_survival():
    pd_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'Survival': item['survival'],
                    'Time': item['times'],
                    'Patients': [item['No'] for i in item['times']]
                }
            ) for item in st.session_state['patients']
        ]
    )
    if st.session_state['display']:
        fig = px.line(pd_data, x="Time", y="Survival", color='Patients', range_y=[0, 1])
    else:
        fig = px.line(pd_data.loc[pd_data['Patients'] == pd_data['Patients'].to_list()[-1], :], x="Time", y="Survival",
                      range_y=[0, 1])
    fig.update_layout(template='simple_white',
                      title={
                          'text': 'Estimated Survival Probability',
                          'y': 0.9,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'font': dict(
                              size=25
                          )
                      },
                      plot_bgcolor="white",
                      xaxis_title="Time, month",
                      yaxis_title="Survival probability",
                      )
    st.plotly_chart(fig, use_container_width=True)


def plot_patients():
    patients = pd.concat(
        [
            pd.DataFrame(
                dict(
                    {
                        'Patients': [item['No']],
                        '3 ans': ["{:.2f}%".format(item['3 ans'] * 100)],
                        '5 ans': ["{:.2f}%".format(item['5 ans'] * 100)],
                        '10 ans': ["{:.2f}%".format(item['10 ans'] * 100)]
                    },
                    **item['arg']
                )
            ) for item in st.session_state['patients']
        ]
    ).reset_index(drop=True)
    st.dataframe(patients)

# @st.cache(show_spinner=True)
def predict():
    print('update patients . ##########')
    print(st.session_state)
    input = []
    for key in input_keys:
        value = st.session_state[key]
        if(key == 'sexe'):
            sex_val =  0 if value == 'Homme' else 1
            input.append(sex_val)
        else :
            input.append(int(value))

    
    input_array = np.array(input).reshape(1, -1)
    input_df = input_array #pd.DataFrame(input_array, columns=input_keys)
    print("Predict input : ",input_array)
    print("Predict input  df : ",input_df)
    survival = mas_model.predict_survival_function(input_df)[0].y
    data = {
        'survival': survival.flatten(),
        'times': [i for i in range(0, len(survival.flatten()))],
        'No': len(st.session_state['patients']) + 1,
        'arg': {key:st.session_state[key] for key in input_keys},
        '3 ans': survival[36],
        '5 ans': survival[60],
        '10 ans': survival[120]
    }
    st.session_state['patients'].append(
        data
    )
    print('update patients ... ##########')

def plot_below_header():
    col1, col2 = st.columns([1, 9])
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        # st.session_state['display'] = ['Single', 'Multiple'].index(
        #     st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
        st.session_state['display'] = ['Single', 'Multiple'].index(
            st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
        # st.radio("Model", ('DeepSurv', 'NMTLR','RSF','CoxPH'), 0,key='model',on_change=predict())
    with col2:
        plot_survival()
    with col4:
        st.metric(
            label='3 years survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['3 ans'] * 100)
        )
    with col5:
        st.metric(
            label='5 years survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['5 ans'] * 100)
        )
    with col6:
        st.metric(
            label='10 years survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['10 ans'] * 100)
        )
    st.write('')
    st.write('')
    st.write('')
    plot_patients()
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

st.header('The prognostic value of machine learning models for CLL', anchor='survival-of-chordoma')
if st.session_state['patients']:
    plot_below_header()
st.subheader("Instructions:")
st.write("1. Select patient's infomation on the left\n2. Press predict button\n3. The model will generate predictions")
st.write('***Note: this model is still a research subject, and the accuracy of the results cannot be guaranteed!***')
st.write("***[Paper link](https://pubmed.ncbi.nlm.nih.gov/)(To be updated)***")
with st.sidebar:
    with st.form("my_form",clear_on_submit = False):
        
         # Slider pour l'âge
        age = st.slider('Âge', min_value=0.0, max_value=100.0, value=70.0, key='age')
        
        # Slider pour le nombre d'anomalies K
        nb_anomalies_k = st.slider('Nb anomalies K', min_value=0.0, max_value=10.0, value=2.0, key='nb_anomalies_k')
        
        anomalie_myc = int(st.checkbox('Anomalie MYC', value=False, key='anomalie_myc'))
        
        # Selectbox pour le sexe
        sexe_option = st.selectbox('Sexe', options=['Homme', 'Femme'], index=1, key='sexe')
        sexe = 0 if sexe_option == 'Homme' else 1
        

    
        gain_2p = int(st.checkbox('Gain 2p', value=False, key='gain_2p'))
        del_17p = int(st.checkbox('Del 17p', value=False, key='del_17p'))
        del_11q = int(st.checkbox('Del 11q', value=False, key='del_11q'))
        tri_12 = int(st.checkbox('Tri 12', value=False, key='tri_12'))
        stade_diag_b_ou_c = int(st.checkbox('Stade diag B ou C', value=False, key='stade_diag_b_ou_c'))
        del_13q = int(st.checkbox('Del 13q', value=False, key='del_13q'))
        del_8p = int(st.checkbox('Del 8p', value=False, key='del_8p'))
        statut_ighv = int(st.checkbox('Statut IGHV', value=False, key='statut_IGHV'))

        #############--------
        
        col8, col9, col10 = st.columns([3, 4, 3])
        with col9:
            prediction = st.form_submit_button(
                'Predict',
                on_click=predict,
                # args=[{key: eval(key.replace(' ', '____')) for key in input_keys}]
            )
