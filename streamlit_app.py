import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import plotly.express as px
from prophet import Prophet

# Se quiser apenas st.line_chart, pode usar matplotlib ou streamlit nativo

st.title("Previsão de Preço do Petróleo (Brent)")

st.write("""
Este aplicativo carrega um modelo *Prophet* previamente treinado (salvo em 
*modelo_prophet.joblib) para prever os próximos **X dias* do preço do petróleo.
""")

# 1) Carregando o modelo Prophet
try:
    modelo_prophet = jl.load('modelo_prophet.joblib')
    st.success("Modelo Prophet carregado com sucesso!")
except FileNotFoundError:
    st.error("Arquivo 'modelo_prophet.joblib' não encontrado! "
             "Por favor, coloque-o na mesma pasta do app.py.")
    st.stop()  # Interrompe a execução se não encontrar o modelo

# 2) Entrada do usuário para horizonte de previsão
horizonte = st.slider(
    "Selecione o horizonte de previsão (em dias):",
    min_value=1,
    max_value=90,
    value=30,
    step=1
)

# 3) Botão para gerar previsão
if st.button("Gerar Previsão"):
    # Cria um DataFrame de datas futuras com base no modelo
    futuro = modelo_prophet.make_future_dataframe(
        periods=horizonte, 
        freq='D'
    )
    
    # Faz a previsão
    forecast = modelo_prophet.predict(futuro)
    
    # Seleciona apenas as linhas referentes ao futuro (últimas 'horizonte' datas)
    forecast_future = forecast.tail(horizonte)
    
    st.subheader("Previsões Geradas")
    st.write("""
    Abaixo estão as previsões para os próximos dias, incluindo o intervalo de 
    confiança de 90% (colunas ⁠ yhat_lower ⁠ e ⁠ yhat_upper ⁠).
    """)
    st.dataframe(forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # Plot do resultado com Plotly
    fig = px.line(
        forecast,
        x='ds',
        y='yhat',
        labels={'ds': 'Data', 'yhat': 'Preço Previsto'},
        title='Previsão do Preço do Petróleo'
    )
    
    # Adicionando banda de confiança (ribbon)
    fig.add_scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'],
        fill=None,
        mode='lines',
        line_color='lightblue',
        name='Limite Inferior'
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill='tonexty',  # preenche até a curva anterior
        mode='lines',
        line_color='lightblue',
        name='Limite Superior'
    )
    
    # Exibe o gráfico
    st.plotly_chart(fig, use_container_width=True)
