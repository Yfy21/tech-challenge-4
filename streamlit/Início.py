import streamlit as st

st.set_page_config(layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] * {
    color: #F5F7FA !important;
}
</style>
""", unsafe_allow_html=True)

st.title('Preditor de Nível de Obesidade')
st.markdown("""
Este aplicativo utiliza um modelo de machine learning treinado com dados demográficos e de hábitos de vida para estimar 
o nível de obesidade de um indivíduo. A classificação é feita em sete categorias, que vão desde *abaixo do peso* 
até *obesidade grau III*.

Use a barra lateral para navegar entre as páginas.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader('Preditor')
    st.markdown("""
Preencha seus dados e receba uma estimativa do seu nível de obesidade com base no modelo treinado.

**Como funciona:**
- Informe suas características demográficas e hábitos de vida no formulário.
- O modelo analisa suas respostas e classifica o resultado em uma das sete categorias de peso.
- Um gráfico SHAP waterfall detalha quais fatores mais influenciaram a predição e em que direção, permitindo entender o raciocínio do modelo para aquela resposta específica.

**Categorias de classificação:**
Abaixo do peso · Peso normal · Sobrepeso I · Sobrepeso II · Obesidade I · Obesidade II · Obesidade III
""")

with col2:
    st.subheader('Dashboard')
    st.markdown("""
Explore os dados utilizados no treinamento do modelo e entenda como ele toma suas decisões.

**Filtros disponíveis**
Refine toda a análise por nível de obesidade, sexo biológico e faixa etária diretamente pela barra lateral.

**O que você encontrará:**
- Indicadores gerais de distribuição da base de dados.
- Distribuição dos respondentes por nível de obesidade.
- Análise estatística por fator de risco — selecione qualquer variável (alimentação, atividade física, meio de transporte etc.) e veja como ela se relaciona com cada nível de obesidade.
- Importância global das variáveis via SHAP — quais fatores mais influenciam o modelo, em média, considerando todos os níveis.
- Impacto por nível de obesidade via SHAP — para uma classe específica, como cada variável contribui positiva ou negativamente para a classificação, com a intensidade do valor codificada em cor.
""")
