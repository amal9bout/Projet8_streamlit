
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import altair as alt
from streamlit_echarts import st_echarts
import pyecharts.options as opts
from pyecharts.charts import Line
from joblib import load
from streamlit_echarts import st_echarts
import plotly.graph_objects as go
import plotly.express as px


# Charger les données
path= "./app_data_top.csv"

app_data_top = pd.read_csv(path)
app_data_top=app_data_top.set_index("SK_ID_CURR")

# Charger le modèle
model = load('lightgbm_model_df1.pkl')


features = ['PAYMENT_RATE', 'EXT_SOURCE_3', 'DAYS_ID_PUBLISH', 'DAYS_BIRTH',
       #'DAYS_REGISTRATION', 'EXT_SOURCE_2', 'DAYS_LAST_PHONE_CHANGE',
       #'AMT_ANNUITY', 'ANNUITY_INCOME_PERC', 'EXT_SOURCE_1', 'DAYS_EMPLOYED',
       #'REGION_POPULATION_RELATIVE', 'INCOME_CREDIT_PERC',
       #'DAYS_EMPLOYED_PERC', 'INCOME_PER_PERSON', 'HOUR_APPR_PROCESS_START',
       #'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_REQ_CREDIT_BUREAU_YEAR',
       #'AMT_INCOME_TOTAL', 'SK_ID_CURR', 
       'score']


# Définir une fonction qui accepte l'ID du client et retourne une prédiction
def predict(client_id):
    try:
        client_id = int(client_id)  # Convertir en entier
        if client_id not in app_data_top.index:
            return "ID client non trouvé"
        score = app_data_top.loc[client_id, "score"]
        return round(100 - score, 2)
    except ValueError:
        return "L'ID client doit être un nombre entier"



def main():
    # Set the app title
    st.header(':red[PAD] bank loan scoring')

    # Add a text input for the user to enter input data
    client_id = st.text_input('Client ID')

    if client_id:
        try:
            SCORE = predict(client_id)
            if isinstance(SCORE, str):
                st.error(SCORE)
            else:
                st.write(f"Votre score est : {SCORE}")
                st.dataframe(app_data_top.loc[int(client_id), features])

                # Afficher le statut du crédit
                credit_status = "crédit accepté" if SCORE >= 50 else "crédit refusé"
                st.write(f"Statut du crédit : {credit_status}")
                
                # Créer le gauge pour afficher le score
                option = {
                    "series": [
                        {
                            "type": "gauge",
                            "startAngle": 180,
                            "endAngle": 0,
                            "min": 0,
                            "max": 100,
                            "center": ["50%", "80%"],
                            'radius': '120%',
                            "splitNumber": 5,
                            "axisLine": {
                                "lineStyle": {
                                    "width": 6,
                                    "color": [
                                        [0.25, "#FF403F"],
                                        [0.5, "#ffa500"],
                                        [0.75, "#FDDD60"],
                                        [1, "#64C88A"],
                                    ],
                                }
                            },
                            "pointer": {
                                "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                                "length": "12%",
                                "width": 30,
                                "offsetCenter": [0, "-60%"],
                                "itemStyle": {"color": "auto"},
                            },
                            "axisTick": {"length": 10, "lineStyle": {"color": "auto", "width": 2}},
                            "splitLine": {"length": 15, "lineStyle": {"color": "auto", "width": 5}},
                            "axisLabel": {
                                "color": "#464646",
                                "fontSize": 12,
                                "distance": -60,
                            },
                            "title": {"offsetCenter": [0, "-20%"], "fontSize": 20},
                            "detail": {
                                "fontSize": 30,
                                "offsetCenter": [0, "0%"],
                                "valueAnimation": True,
                                "color": "auto",
                                "formatter": SCORE,
                            },
                            "data": [{"value": SCORE, "name": "Client score"}],
                        }
                    ]
                }

                st_echarts(option, width="450px", height="350px", key="gauge")

        except ValueError:
            st.error("L'ID client doit être un nombre entier.")
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

    # Afficher les scores des caractéristiques
    if st.button("FEATURES"):
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Features global", "Features detail", "Feature selection", "Bivariate analysis"])
        
        with tab1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=features, y=app_data_top[features].mean().tolist(), name='MEAN',
                                      line=dict(color='firebrick', width=4)))
            fig1.add_trace(go.Scatter(x=features, y=app_data_top[features].max().tolist(), name='MAX',
                                      line=dict(color='royalblue', width=4)))
            fig1.add_trace(go.Scatter(x=features, y=app_data_top[features].min().tolist(), name='MIN',
                                      line=dict(color='green', width=4)))
            fig1.update_layout(title='Features score',
                               xaxis_title="Features",
                               yaxis_title='Values')
            st.plotly_chart(fig1)

        with tab2:
            fig2 = px.box(app_data_top[features], x=features, points="all")
            st.plotly_chart(fig2)

        with tab3:
            selected_feature = st.selectbox('Select a feature', features)
            fig3 = px.histogram(app_data_top, x=selected_feature, marginal="box", nbins=30, title=f'Distribution of {selected_feature}')
            fig3.add_vline(x=app_data_top.loc[int(client_id), selected_feature], line_dash="dash", line_color="red")
            st.plotly_chart(fig3)

            fig4 = px.box(app_data_top, x=selected_feature, title=f'Box plot of {selected_feature}')
            fig4.add_vline(x=app_data_top.loc[int(client_id), selected_feature], line_dash="dash", line_color="red")
            st.plotly_chart(fig4)
        
        with tab4:
            feature_x = st.selectbox('Select feature for X-axis', features, index=0, key='feature_x')
            feature_y = st.selectbox('Select feature for Y-axis', features, index=1, key='feature_y')
            fig5 = px.scatter(app_data_top, x=feature_x, y=feature_y, color='score',
                              title=f'Bivariate analysis between {feature_x} and {feature_y}',
                              color_continuous_scale='Viridis')
            fig5.add_scatter(x=[app_data_top.loc[int(client_id), feature_x]],
                             y=[app_data_top.loc[int(client_id), feature_y]],
                             mode='markers',
                             marker=dict(size=12, color='red', symbol='x'),
                             name='Client Position')
            st.plotly_chart(fig5)

if __name__ == '__main__':
    main()