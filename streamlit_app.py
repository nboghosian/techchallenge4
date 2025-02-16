import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import plotly.graph_objects as go
from prophet import Prophet

st.title("🛢️ Previsão de Preço do Petróleo (Brent)")

st.write("""
Este aplicativo carrega um modelo *Prophet* previamente treinado para prever 
os próximos dias do preço do petróleo e mostra o histórico em uma cor e 
a parte futura em outra.
""")

# 1) Carregando o modelo Prophet
try:
    modelo_prophet = jl.load('modelo_prophet.joblib')
    st.success("Modelo Prophet carregado com sucesso!")
except FileNotFoundError:
    st.error("Arquivo 'modelo_prophet.joblib' não encontrado! "
             "Por favor, coloque-o na mesma pasta do app.py.")
    st.stop()

# 2) Selecionar horizonte de previsão
horizonte = st.slider(
    "Selecione o horizonte de previsão (em dias):",
    min_value=1,
    max_value=90,
    value=30,
    step=1
)

# 3) Ao clicar em "Gerar Previsão"
if st.button("Gerar Previsão"):
    # Gera todo o DataFrame de previsão (histórico + futuro)
    futuro = modelo_prophet.make_future_dataframe(periods=horizonte, freq='D')
    forecast = modelo_prophet.predict(futuro)

    # Separa em duas partes:
    # - df_history: tudo exceto os últimos `horizonte` dias
    # - df_future: últimos `horizonte` dias (o "trecho previsto")
    total_rows = forecast.shape[0]
    df_history = forecast.iloc[: total_rows - horizonte]
    df_future = forecast.iloc[total_rows - horizonte : ]

    st.subheader("Previsões Geradas (Últimos dias)")
    st.write("""
    Abaixo estão **somente** as previsões futuras (até o horizonte selecionado),
    incluindo o intervalo de confiança.
    """)
    st.dataframe(df_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # 4) Construindo o gráfico com plotly.graph_objects
    fig = go.Figure()

    # --- HISTÓRICO / TREINO ---
    fig.add_trace(go.Scatter(
        x=df_history['ds'], 
        y=df_history['yhat'],
        mode='lines',
        name='Histórico (ajuste do modelo)',
        line=dict(color='blue')
    ))
    # Intervalos do histórico (opcional)
    fig.add_trace(go.Scatter(
        x=df_history['ds'],
        y=df_history['yhat_lower'],
        fill=None,
        mode='lines',
        line_color='lightblue',
        name='Limite Inferior Histórico'
    ))
    fig.add_trace(go.Scatter(
        x=df_history['ds'],
        y=df_history['yhat_upper'],
        fill='tonexty',
        mode='lines',
        line_color='lightblue',
        name='Limite Superior Histórico'
    ))

    # --- FUTURO / PREVISÃO ---
    fig.add_trace(go.Scatter(
        x=df_future['ds'], 
        y=df_future['yhat'],
        mode='lines',
        name='Previsão',
        line=dict(color='red')
    ))
    # Intervalos do futuro
    fig.add_trace(go.Scatter(
        x=df_future['ds'],
        y=df_future['yhat_lower'],
        fill=None,
        mode='lines',
        line_color='pink',
        name='Limite Inferior Futuro'
    ))
    fig.add_trace(go.Scatter(
        x=df_future['ds'],
        y=df_future['yhat_upper'],
        fill='tonexty',
        mode='lines',
        line_color='pink',
        name='Limite Superior Futuro'
    ))

    fig.update_layout(
        title='Previsão do Preço do Petróleo (Histórico vs. Futuro)',
        xaxis_title='Data',
        yaxis_title='Preço Previsto'
    )

    st.plotly_chart(fig, use_container_width=True)

