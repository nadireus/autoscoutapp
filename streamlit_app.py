import streamlit as st
import pickle
import pandas as pd



st.sidebar.title('Auto Preis Vorhersage')

html_temp = """
<div style="background-color:pink;padding:15px">
<h2 style="color:black;text-align:center;">Welches Auto ist günstig?</h2><h1>

</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


age = st.sidebar.selectbox("Wie alt ist Ihr Auto?", (1,2,3,4,5,6,7,8,9))
hp = st.sidebar.slider("Was ist die PS Ihres Autos?",60,200,step=5)
km = st.sidebar.slider("Was ist die km von Ihrem Auto?",0,100000,step=500)
gear_type = st.sidebar.radio("Wählen Sie der Getriebetyp Ihres Autos?",('Automatic','Manual','Semi-automatic'))
car_model = st.sidebar.selectbox("Wählen Sie Ihr Auto-Modell ?",['A1','A2','A3','Astra','Clio','Corsa','Espace','Insignia'])




model_name=st.selectbox('Laut Angaben der Experten des CAR-Centers der Universität Duisburg-Essen ist der Durchschnittspreis für Neuwagen im Zeitraum von 2005 bis 2015 um etwa 6.000 Euro angestiegen.Im ersten Halbjahr 2015 gaben Neuwagenkäufer rund 28.000 Euro für ein fabrikneues Modell aus.Doch welche Marken bieten überhaupt preiswerte Autos an? Dieses App helfen Ihnen dabei, den Autokauf günstig abzuwickeln.Wählen  Sie  eine  ML Modell und geben sie die benötige Informationen ein.Erfahren Sie der ungefähre Preise des Autos ',('Xgboost','RanFor'))
if model_name=='Xgboost':
    model=pickle.load(open('xgb_model','rb'))
    st.success(' Ihnen ausgewählten Modell ist  {} '.format(model_name))
elif model_name=='RanFor':
    model=pickle.load(open('rf_model','rb'))
    st.success(' Ihnen ausgewählten Modell ist {} '.format(model_name))

my_dict={

    'age': age,
    'hp':hp,
    'km':km,
    'model':car_model,
    'gearing_type':gear_type
    
}

df = pd.DataFrame.from_dict([my_dict])

columns=pickle.load(open('my_columns','rb'))

df=pd.get_dummies(df).reindex(columns=columns, fill_value=0)

st.header('Nachfolgend ist die Konfiguration Ihres Autos:')
st.table(df)


st.subheader('Drücken Sie die Vorhersage-Taste, wenn Configuration ist okay')
if st.button('Voraussagen'):
    if model_name=='RanFor':
        scaler=pickle.load(open('my_scaler','rb'))
        df=scaler.transform(df)
        prediction=model.predict(df)
    else:
        prediction=model.predict(df)
    st.success('Die Schätzung Ihres Modells ist €{}'.format(int(prediction[0])))
    st.balloons()