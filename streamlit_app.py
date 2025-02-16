import streamlit as st
import pandas as pd
import joblib as jl
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet

# Título geral do app
st.title("🛢️ Análise e Previsão de Preço do Petróleo (Brent)")

# =========================================
# CRIA AS DUAS ABAS
# =========================================
tab1, tab2 = st.tabs(["Contextualização", "Previsões"])

# -------------------------------------------
# TAB 1: Texto e Imagens
# -------------------------------------------
with tab1:
    st.header("Entenda o Contexto do Petróleo Brent")
    st.write("""
    O mercado de petróleo é um dos mais influentes na economia global, impactando desde o custo de produção industrial até os preços ao consumidor. O petróleo Brent, referência internacional para precificação da commodity, é negociado diariamente e sua volatilidade pode ser influenciada por fatores geopolíticos, variações na demanda, mudanças na oferta e políticas econômicas (Hamilton, 2009).
A análise de dados históricos de preços do petróleo Brent, disponível no repositório do Instituto de Pesquisa Econômica Aplicada (IPEA), fornece uma base essencial para identificar tendências, padrões sazonais e possíveis ciclos de preço. Essa base de dados é composta por duas colunas principais: data e preço (em dólares), permitindo uma abordagem quantitativa para modelagem preditiva e análise de impacto econômico (IPEA, 2024).
Neste contexto, a exploração desses dados pode oferecer insights estratégicos para investidores, gestores públicos e empresas do setor energético, possibilitando a construção de modelos preditivos e paineis interativos que auxiliam na tomada de decisão (Baumeister & Kilian, 2016).

**Referências**
- Baumeister, C., & Kilian, L. (2016). Forty years of oil price fluctuations: Why the price of oil may still surprise us. Journal of Economic Perspectives, 30(1), 139-160.
- Hamilton, J. D. (2009). Causes and Consequences of the Oil Shock of 2007-08. Brookings Papers on Economic Activity, 2009(1), 215-261.
- Instituto de Pesquisa Econômica Aplicada (IPEA). (2024). Base de dados histórica do preço do petróleo Brent. Disponível em: www.ipea.gov.br (Acessado em: [data de acesso]).

    """)


    # st.image("minha_imagem.png", caption="Exemplo de imagem local") 
    st.image("https://s2.glbimg.com/ZIPcGot1Af66bTwWlLN0CT1U6FM=/620x350/e.glbimg.com/og/ed/f/original/2020/07/01/111245902_gettyimages-103256923.jpg", caption="")


    st.subheader("Histórico de Preços com Filtro de Datas")

    # Função para carregar os dados com cache
    @st.cache_data
    def carregar_dados():
        return pd.read_csv("petroleo_hist.csv", sep=";", parse_dates=["ds"])

    df = carregar_dados()

    # Obtém a data mínima e máxima do DataFrame
    data_min = df["ds"].min()
    data_max = df["ds"].max()

    st.write("Selecione o intervalo de datas que deseja visualizar:")

    # Widget para selecionar intervalo de datas
    intervalo_datas = st.date_input(
        "Intervalo",
        value=[data_min, data_max],
        min_value=data_min,
        max_value=data_max
    )

    # Cria duas colunas para os botões
    col1, col2 = st.columns(2)

    with col1:
        aplicar = st.button("Aplicar Filtro")
    with col2:
        limpar = st.button("Limpar Filtro")

    # Se o usuário clicar em "Aplicar Filtro"
    if aplicar:
        if len(intervalo_datas) == 2:
            data_inicial, data_final = intervalo_datas
        else:
            data_inicial, data_final = data_min, data_max

        # Filtra o DataFrame conforme as datas selecionadas
        df_filtrado = df[(df["ds"] >= pd.to_datetime(data_inicial)) &
                         (df["ds"] <= pd.to_datetime(data_final))]

        st.write(f"Exibindo dados de {data_inicial} até {data_final}")
        fig = px.line(
            df_filtrado,
            x="ds",
            y="y",  # ajuste o nome da coluna se necessário
            title="Histórico de Preços do Petróleo Brent",
            labels={"ds": "Data", "y": "Preço (US$)"}
        )
        fig.update_traces(line=dict(color="lightblue"))
        st.plotly_chart(fig, use_container_width=True)

    # Se o usuário clicar em "Limpar Filtro"
    if limpar:
        st.write("Exibindo dados completos")
        fig = px.line(
            df,
            x="ds",
            y="y",
            title="Histórico de Preços do Petróleo Brent (Completo)",
            labels={"ds": "Data", "y": "Preço (US$)"}
        )
        st.plotly_chart(fig, use_container_width=True)



# -------------------------------------------
# TAB 2: Forecast com Prophet
# -------------------------------------------
with tab2:
    st.header("Previsão do Preço com Prophet")

    st.write("""
Este aplicativo carrega um modelo *Prophet* previamente treinado para prever os próximos dias do preço do petróleo. 
O gráfico mostrará o histórico (apenas 1 ano antes do início da previsão) em azul e a previsão futura em vermelho.
    """)

    # 1) Carregando o modelo Prophet
    try:
        modelo_prophet = jl.load('modelo_prophet.joblib')
        st.success("Modelo Prophet carregado com sucesso!")
    except FileNotFoundError:
        st.error("Arquivo 'modelo_prophet.joblib' não encontrado! Por favor, coloque-o na mesma pasta do app.py.")
        st.stop()

    # 2) Selecionar horizonte de previsão
    horizonte = st.slider(
        "Selecione o horizonte de previsão (em dias):",
        min_value=1,
        max_value=90,
        value=30,
        step=1
    )

    # 3) Botão para gerar previsão
    if st.button("Gerar Previsão"):
        # Gera todo o DataFrame de previsão (historico + futuro)
        futuro = modelo_prophet.make_future_dataframe(periods=horizonte, freq='D')
        forecast = modelo_prophet.predict(futuro)

        # Separa os dados futuros: os últimos 'horizonte' dias
        df_future = forecast.iloc[-horizonte:]
    
        # Define a data de início da previsão futura
        forecast_future_start = df_future['ds'].iloc[0]
        # Calcula a data correspondente a 1 ano antes
        one_year_before = forecast_future_start - pd.DateOffset(years=1)
    
    # Seleciona a parte histórica: dados entre one_year_before e o início do forecast futuro
        df_history = forecast[(forecast['ds'] >= one_year_before) & (forecast['ds'] < forecast_future_start)]
    
        st.subheader("Previsões Geradas")
        st.write(f"Exibindo histórico de 1 ano (a partir de {one_year_before.date()}) até o início da previsão ({forecast_future_start.date()}) e a previsão para os próximos {horizonte} dia(s).")
        st.dataframe(df_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # Construindo o gráfico com Plotly Graph Objects
        fig = go.Figure()
    
    # Histórico: linha em azul
        fig.add_trace(go.Scatter(
            x=df_history['ds'],
            y=df_history['yhat'],
            mode='lines',
            name='Histórico (1 ano)',
            line=dict(color='blue')
        ))
    # Limites do histórico (opcional)
        fig.add_trace(go.Scatter(
            x=df_history['ds'],
            y=df_history['yhat_lower'],
            mode='lines',
            name='Limite Inferior (Hist)',
            line=dict(color='lightblue'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_history['ds'],
            y=df_history['yhat_upper'],
            mode='lines',
            name='Limite Superior (Hist)',
            line=dict(color='lightblue'),
            fill='tonexty',
            showlegend=False
        ))
    
        # Previsão futura: linha em vermelho
        fig.add_trace(go.Scatter(
            x=df_future['ds'],
            y=df_future['yhat'],
            mode='lines',
            name='Previsão Futura',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=df_future['ds'],
            y=df_future['yhat_lower'],
            mode='lines',
            name='Limite Inferior (Fut)',
            line=dict(color='pink'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_future['ds'],
            y=df_future['yhat_upper'],
            mode='lines',
            name='Limite Superior (Fut)',
            line=dict(color='pink'),
            fill='tonexty',
            showlegend=False
        ))
    
        fig.update_layout(
            title='Previsão do Preço do Petróleo: Histórico (1 ano) vs. Futuro',
            xaxis_title='Data',
            yaxis_title='Preço Previsto'
        )
    
        st.plotly_chart(fig, use_container_width=True)



