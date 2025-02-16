import streamlit as st
import pandas as pd
import joblib as jl
import plotly.graph_objects as go
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

    st.subheader("Imagem Exemplo")
    st.write("Podemos mostrar gráficos estatísticos, diagramas, ou qualquer figura explicativa.")
    # Se quiser inserir uma imagem local ou de URL
    # st.image("minha_imagem.png", caption="Exemplo de imagem local") 
    # ou
    # st.image("https://path.to/alguma_imagem.jpg", caption="Imagem de contexto")

    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.title("Gráfico Histórico com Filtro de Datas")

    # 1) Ler o CSV (exemplo: "historico_brent.csv") na mesma pasta do app
    @st.cache_data  # cache para acelerar re-leituras
    def carregar_dados():
        df = pd.read_csv("petroleo_hist.csv", parse_dates=["ds"])
        return df

    df = carregar_dados()

    # Mostra o DataFrame inteiro (opcional)
    st.write("Dados Históricos (primeiras linhas):")
    st.dataframe(df.head())

    # 2) Selecionar intervalo de datas
    # Pega data mínima e máxima do próprio DataFrame
    data_min = df["ds"].min()
    data_max = df["ds"].max()

    st.write("Selecione o intervalo de datas que deseja visualizar:")
    intervalo_datas = st.date_input("Intervalo", 
                                value=[data_min, data_max], 
                                min_value=data_min, 
                                max_value=data_max)

    # O Streamlit retorna uma tupla ou lista [start_date, end_date]
    if len(intervalo_datas) == 2:
        data_inicial, data_final = intervalo_datas[0], intervalo_datas[1]
    else:
        data_inicial, data_final = data_min, data_max

    # 3) Filtrando o DataFrame
    df_filtrado = df[(df["ds"] >= pd.to_datetime(data_inicial)) & 
                 (df["ds"] <= pd.to_datetime(data_final))]

    st.write(f"Exibindo dados de {data_inicial} até {data_final}")

    # 4) Plotar o resultado
    fig = px.line(df_filtrado, 
              x="ds", 
              y="y",  # Ajuste com a coluna de preço
              title="Histórico de Preços do Petróleo Brent",
              labels={"ds": "Data", "y": "Preço (US$)"})

    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------
# TAB 2: Forecast com Prophet
# -------------------------------------------
with tab2:
    st.header("Previsão do Preço com Prophet")

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

    # 3) Botão para gerar previsão
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
        st.write("Exibimos apenas as **previsões futuras** (até o horizonte selecionado).")
        st.dataframe(df_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Construindo o gráfico
        fig = go.Figure()

        # HISTÓRICO
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
            name='Limite Inferior (Hist)'
        ))
        fig.add_trace(go.Scatter(
            x=df_history['ds'],
            y=df_history['yhat_upper'],
            fill='tonexty',
            mode='lines',
            line_color='lightblue',
            name='Limite Superior (Hist)'
        ))

        # FUTURO
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
            fill=None,
            mode='lines',
            line_color='pink',
            name='Limite Inferior (Fut)'
        ))
        fig.add_trace(go.Scatter(
            x=df_future['ds'],
            y=df_future['yhat_upper'],
            fill='tonexty',
            mode='lines',
            line_color='pink',
            name='Limite Superior (Fut)'
        ))

        fig.update_layout(
            title='Previsão do Preço do Petróleo (Histórico vs. Futuro)',
            xaxis_title='Data',
            yaxis_title='Preço Previsto'
        )

        st.plotly_chart(fig, use_container_width=True)



