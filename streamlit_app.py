import streamlit as st
import pandas as pd
import joblib as jl
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet

# Título geral do app
st.title("🛢️ Análise e Previsão de Preço do Petróleo (Brent)")

# =========================================
# CRIA AS ABAS
# =========================================
tab1, tab2, tab3, tab4 = st.tabs(["Contextualização", "Insights", "Previsão", "Plano de Deploy"])

# -------------------------------------------
# TAB 1: Texto e Imagens
# -------------------------------------------
with tab1:
    st.header("💬 Entenda o Contexto do Petróleo Brent")
    st.write("""
    O mercado de petróleo é um dos mais influentes na economia global, impactando desde o custo de produção industrial até os preços ao consumidor. O petróleo Brent, referência internacional para precificação da commodity, é negociado diariamente e sua volatilidade pode ser influenciada por fatores geopolíticos, variações na demanda, mudanças na oferta e políticas econômicas (Hamilton, 2009).
A análise de dados históricos de preços do petróleo Brent, disponível no repositório do Instituto de Pesquisa Econômica Aplicada (IPEA), fornece uma base essencial para identificar tendências, padrões sazonais e possíveis ciclos de preço. Essa base de dados é composta por duas colunas principais: data e preço (em dólares), permitindo uma abordagem quantitativa para modelagem preditiva e análise de impacto econômico (IPEA, 2024).
Neste contexto, a exploração desses dados pode oferecer insights estratégicos por meio da construção de modelos preditivos e painéis que auxiliem na tomada de decisão.

**Referências**
- Hamilton, J. D. (2009). Causes and Consequences of the Oil Shock of 2007-08. Brookings Papers on Economic Activity, 2009(1), 215-261.
- Instituto de Pesquisa Econômica Aplicada (IPEA). (2024). Base de dados histórica do preço do petróleo Brent. Disponível em: www.ipea.gov.br.

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
# TAB 2: Insights
# -------------------------------------------
with tab2:
    st.header("✨ Principais Insights")

    st.write("""
    A fim de exibir os principais insights e um dashboard interativo foi desenvolvido utilizando o Power BI.
    """)

    st.image("petroleo1.png", caption="Página Inicial")
    
    st.image("petroleo2.png", caption="1998")
    
    st.image("petroleo3.png", caption="2008")
    
    st.image("petroleo4.png", caption="2020")
    
    st.image("petroleo5.png", caption="2022")
    

    st.markdown("Baixe o [arquivo .pbix no GitHub](https://github.com/nboghosian/techchallenge4_powerbi) e interaja com os visuais!")



    
    
# -------------------------------------------
# TAB 3: Forecast com Prophet
# -------------------------------------------
with tab3:
    st.header("💰 Previsão do Preço com Prophet")
    
    st.write("""
    Apesar da volatilidade inerente ao mercado de petróleo, as análises iniciais de autocorrelação indicaram a presença de memória longa na série, sugerindo que valores passados exercem influência ao longo de períodos mais extensos. Esse comportamento pode estar ligado a ciclos prolongados de oferta e demanda, à ação coordenada de grandes exportadores ou a flutuações econômicas persistentes.

Com base nesses achados, decidiu-se testar diferentes modelos:

- Naive e SeasonalNaive: Servem como pontos de partida e linhas de base para comparação. O Naive assume que o próximo valor será igual ao último observado, enquanto o SeasonalNaive introduz o conceito de sazonalidade, projetando o valor de hoje com base no mesmo dia de um período anterior (por exemplo, o valor observado 7 dias atrás em dados diários).

- Prophet: Desenvolvido pelo Facebook (Meta), lida bem com tendências não lineares, sazonalidades múltiplas e efeitos de feriados ou eventos específicos, caso sejam informados. Também pode capturar parte de uma memória mais longa, sobretudo quando há sinais de ciclos prolongados.

A decisão sobre qual modelo utilizar depende, em última instância, dos resultados de validação e da capacidade de cada método em capturar tanto as flutuações de curto prazo quanto a persistência de longo prazo observada na série. A presença de memória longa levanta a possibilidade de que modelos como o Prophet, quando configurados para lidar com sazonalidades ou componentes de tendência mais extensas, possam apresentar melhor desempenho no horizonte pretendido. Por outro lado, métodos mais simples como Naive e SeasonalNaive servem de benchmarks e podem, surpreendentemente, apresentar bons resultados em cenários de alta variabilidade onde grande parte do comportamento recente determina o valor futuro imediato.
Dado o contexto da consultoria, em que decisões estratégicas costumam se estender além de poucos dias, é mais vantajoso focar em horizontes de previsão mais amplos (30 e 90 dias). Embora o modelo Naive tenha apresentado bons resultados de curtíssimo prazo (1 e 7 dias), ele se mostra limitado quando a projeção é estendida. Já o Prophet demonstrou maior precisão ao longo de períodos maiores, capturando melhor tendências e sazonalidades que impactam o mercado em semanas ou meses.

Portanto, visando orientar o cliente sobre planejamento, compra de insumos, definição de estoques ou negociações de médio e longo prazo, o Prophet foi a escolha mais indicada, visto que apresentou menor erro (WMAPE) para janelas de 30 e 90 dias. Em suma, a capacidade do Prophet de modelar componentes de tendência e sazonalidade faz com que ele ofereça uma previsão mais robusta e alinhada às necessidades estratégicas típicas de uma consultoria voltada a decisões que excedem poucos dias de antecipação.

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
        st.dataframe(df_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True))
    
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

    st.markdown("A análise exploratória e desenvolvimento do modelo estão disponíveis no [Google Colab](https://colab.research.google.com/drive/1h8VVlJ512b_nhdhQiQ5LGYupaQheOpUQ?usp=sharing).")

# -------------------------------------------
# TAB 4: Plano de Deploy
# -------------------------------------------
with tab4:

    st.header("🎖️ Plano Completo de Deploy")

    st.write(""" 1. **Desenvolvimento e Testes Locais**  
   - Criar e testar o app localmente, garantindo que todas as funcionalidades estejam funcionando conforme o esperado.  
   - Utilizar um ambiente virtual (venv ou conda) e gerenciar as dependências (via requirements.txt).

2. **Gerenciamento de Dependências e Versionamento**  
   - Certificar-se de que todas as bibliotecas necessárias estejam listadas em um arquivo requirements.txt.  
   - Versionar o código usando Git, hospedando-o em um repositório (GitHub, GitLab, etc.) para facilitar a integração contínua.

3. **Configuração do Ambiente de Produção**  
   - Configurar variáveis de ambiente, chaves e outras configurações sensíveis de forma segura.  
   - Preparar um arquivo de configuração (por exemplo, .env) para gerenciar essas variáveis, se necessário.

4. **Escolher da Plataforma de Deploy**  
   - Utilizar o Streamlit Cloud, que é uma plataforma gratuita e simples para deploy de apps Streamlit.  
   - Ou optar por outras plataformas, como Heroku, Render, AWS ou GCP, se houver necessidade de mais controle ou escalabilidade.

5. **Deploy**  
   - No caso do Streamlit Cloud, conectar o repositório e seguir as instruções da plataforma para realizar o deploy.  
   - Para outras plataformas, fazer o build do contêiner (se aplicável) e configurar o processo de deploy (por exemplo, realizar o push para o serviço, configurar o domínio, etc.).

6. **Monitoramento e Manutenção**  
   - Após o deploy, monitorar o app quanto à performance e erros.  
   - Configurar logs e, se possível, alertas para identificar problemas rapidamente.  
   - Planejar atualizações e retreinamento do modelo, se for o caso, e implementar um processo de CI/CD para facilitar o deploy de novas versões.

 """)


