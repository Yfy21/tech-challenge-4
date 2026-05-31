import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] * {
    color: #F5F7FA !important;
}
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div,
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button {
    background-color: #F5F7FA !important;
    border-color: rgba(31,58,95,0.3) !important;
}
[data-testid="stFormSubmitButton"] button {
    background-color: #1F3A5F !important;
    color: #F5F7FA !important;
    border-color: #1F3A5F !important;
}
</style>
""", unsafe_allow_html=True)

API_URL = os.environ.get('API_URL', 'http://localhost:5000')
MODEL_DIR = os.environ.get('MODEL_DIR', './model')

FEATURE_NAME_PT = {
    'num__age': 'Idade',
    'ordinal_cat_num__fcvc': 'Consumo de vegetais',
    'ordinal_cat_num__ncp': 'Nº de refeições',
    'ordinal_cat_num__ch2o': 'Consumo de água',
    'ordinal_cat_num__faf': 'Atividade física',
    'ordinal_cat_num__tue': 'Tempo de tela',
    'ordinal_cat_text__caec': 'Come entre refeições',
    'ordinal_cat_text__calc': 'Consumo de álcool',
    'ordinal_cat_text__mtrans': 'Meio de transporte',
    'remainder__gender': 'Sexo biológico',
    'remainder__family_history': 'Histórico familiar',
    'remainder__favc': 'Alimentos calóricos',
    'remainder__smoke': 'Fumante',
    'remainder__scc': 'Monitora calorias',
}

OBESITY_PT = {
    'Insufficient_Weight': 'Abaixo do peso',
    'Normal_Weight': 'Peso normal',
    'Overweight_Level_I': 'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I': 'Obesidade I',
    'Obesity_Type_II': 'Obesidade II',
    'Obesity_Type_III': 'Obesidade III',
}

@st.cache_resource
def load_model_and_explainer():
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    return model, explainer

# Elementos da página - Título e Subtítulo
st.title('Preditor de Obesidade com Base em Fatores de Risco')
st.write('Preencha suas informações para saber se seus hábitos implicam em risco de obesidade.')

# Elementos da página - Formulário
with st.form('prediction_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Qual é o seu sexo biológico?',
                              ['Masculino', 'Feminino'])
        age = st.number_input('Qual é a sua idade?',
                              min_value=0, max_value=123, value=25)
        family_history = st.selectbox('Possui histórico familiar de sobrepeso?',
                                      ['Sim', 'Não'])
        smoke = st.selectbox('É fumante?',
                             ['Sim', 'Não'])
        mtrans = st.selectbox('Qual é o seu principal meio de transporte?',
                              ['Carro', 'Moto', 'Bicicleta', 'Transporte público', 'A pé'])

    with col2:
        ncp = st.selectbox('Quantas refeições você faz por dia? ("lanchinhos" não contam)',
                           ['1', '2', '3', '4 ou mais'])
        caec = st.selectbox('Costuma comer entre as refeições?',
                            ['Não', 'Às vezes', 'Sim, frequentemente', 'Sim, sempre'])
        scc = st.selectbox('Você monitora sua ingestão diária de calorias?',
                           ['Sim', 'Não'])
        fcvc = st.selectbox('Com que frequência você consome vegetais?',
                            ['Raramente', 'Às vezes', 'Sempre'])
        favc = st.selectbox('Você consome alimentos altamente calóricos com frequência? (Ex.: fritura, pizzas, refrigerantes, doces, etc.)',
                            ['Sim', 'Não'])

    with col3:
        faf = st.selectbox('Com que frequência você faz atividade física?',
                           ['Nunca', 'De 1 a 2x na semana', 'De 3 a 4x na semana', '5x por semana ou mais'])
        tue = st.selectbox('Qual é o seu tempo de tela diário?',
                           ['Até 2h por dia', 'De 3 a 5h por dia', 'Mais de 5h por dia'])
        calc = st.selectbox('Com que frequência você consome bebidas alcoólicas?',
                            ['Nunca', 'Às vezes', 'Frequentemente', 'Sempre'])
        ch2o = st.selectbox('Qual é, em média, o seu consumo diário de água?',
                            ['Menos de 1L por dia', 'De 1L a 2L por dia', 'Mais de 2L por dia'])

    submitted = st.form_submit_button('Veja o seu resultado')

# Mapas de conversão de respostas
fcvc_map = {'Raramente': 1, 'Às vezes': 2, 'Sempre': 3}
ncp_map = {'1': 1, '2': 2, '3': 3, '4 ou mais': 4}
caec_map = {'Não': 'No', 'Às vezes': 'Sometimes', 'Sim, frequentemente': 'Frequently', 'Sim, sempre': 'Always'}
ch2o_map = {'Menos de 1L por dia': 1, 'De 1L a 2L por dia': 2, 'Mais de 2L por dia': 3}
faf_map = {'Nunca': 0, 'De 1 a 2x na semana': 1, 'De 3 a 4x na semana': 2, '5x por semana ou mais': 3}
tue_map = {'Até 2h por dia': 0, 'De 3 a 5h por dia': 1, 'Mais de 5h por dia': 2}
calc_map = {'Nunca': 'No', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
mtrans_map = {'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike',
              'Transporte público': 'Public_Transportation', 'A pé': 'Walking'}

obesity_result_map = {
    'Insufficient_Weight': 'pessoas abaixo do peso',
    'Normal_Weight': 'peso normal',
    'Overweight_Level_I': 'sobrepeso nível I',
    'Overweight_Level_II': 'sobrepeso nível II',
    'Obesity_Type_I': 'obesidade nível I',
    'Obesity_Type_II': 'obesidade nível II',
    'Obesity_Type_III': 'obesidade nível III',
}

# Geração da resposta
if submitted:
    payload = {
        'gender': gender == 'Feminino',
        'age': age,
        'family_history': family_history == 'Sim',
        'favc': favc == 'Sim',
        'fcvc': fcvc_map[fcvc],
        'ncp': ncp_map[ncp],
        'caec': caec_map[caec],
        'smoke': smoke == 'Sim',
        'ch2o': ch2o_map[ch2o],
        'scc': scc == 'Sim',
        'faf': faf_map[faf],
        'tue': tue_map[tue],
        'calc': calc_map[calc],
        'mtrans': mtrans_map[mtrans],
    }

    response = requests.post(f'{API_URL}/predict', json=payload)
    prediction = response.json()['prediction']
    if prediction == 'Normal_Weight':
        st.success(f'Seus hábitos estão associados a **{obesity_result_map[prediction]}**.')
    elif prediction in ['Insufficient_Weight', 'Overweight_Level_I', 'Overweight_Level_II']:
        st.warning(f'Seus hábitos estão associados a **{obesity_result_map[prediction]}**.')
    else:
        st.error(f'Seus hábitos estão associados a **{obesity_result_map[prediction]}**.')

    st.divider()
    st.subheader('O que influenciou este resultado?')
    st.caption(f'Classificação: **{OBESITY_PT[prediction]}**. Entenda o quanto a resposta que você deu para cada questão contribui para a sua classificação.')
    st.caption('Obs.: Barras amarelas aumentam a probabilidade de pertencer a essa classe; barras azuis reduzem.')
    try:
        model, explainer = load_model_and_explainer()
        preprocessor = model.named_steps['preprocessor']
        selector = model.named_steps.get('selector')
        classifier = model.named_steps['classifier']

        input_df = pd.DataFrame([payload])
        X_t = preprocessor.transform(input_df)
        feat_names = preprocessor.get_feature_names_out()
        if selector is not None:
            X_t = selector.transform(X_t)
            feat_names = feat_names[selector.get_support()]

        shap_exp = explainer(X_t)
        classes = list(classifier.classes_)
        class_idx = classes.index(prediction)

        if shap_exp.values.ndim == 3:
            exp_class = shap.Explanation(
                values=shap_exp.values[0, :, class_idx],
                base_values=shap_exp.base_values[0, class_idx],
                data=shap_exp.data[0],
                feature_names=[FEATURE_NAME_PT.get(f, f) for f in feat_names],
            )
        else:
            exp_class = shap.Explanation(
                values=shap_exp.values[0],
                base_values=shap_exp.base_values[0],
                data=shap_exp.data[0],
                feature_names=[FEATURE_NAME_PT.get(f, f) for f in feat_names],
            )

        values = exp_class.values * 100
        base_val = float(exp_class.base_values) * 100
        f_x = base_val + float(values.sum())

        # Sort descending by |SHAP| — most impactful at top
        sorted_idx = np.argsort(np.abs(values))[::-1]
        sorted_values = values[sorted_idx]
        sorted_names = [exp_class.feature_names[i] for i in sorted_idx]

        # Bar base = running cumulative total before each feature is added
        bar_bases = base_val + np.concatenate([[0], np.cumsum(sorted_values[:-1])])
        bar_colors = ['#E0A458' if v > 0 else '#1F3A5F' for v in sorted_values]

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(
            orientation='h',
            y=sorted_names,
            x=sorted_values,
            base=bar_bases,
            marker=dict(color=bar_colors),
            text=[f'{v:+.2f}%' for v in sorted_values],
            textposition='outside',
            textfont=dict(color='#1F3A5F'),
            showlegend=False,
        ))

        fig_wf.add_vline(
            x=base_val,
            line_dash='dash', line_color='rgba(31,58,95,0.4)', line_width=1.5,
            annotation_text=f'Prob. média = {base_val:.2f}%',
            annotation_position='bottom right',
            annotation_font=dict(color='#1F3A5F', size=11),
        )
        fig_wf.add_vline(
            x=f_x,
            line_dash='solid', line_color='#2F6F73', line_width=2,
            annotation_text=f'Sua probabilidade = {f_x:.2f}%',
            annotation_position='top right',
            annotation_font=dict(color='#2F6F73', size=11),
        )

        fig_wf.update_layout(
            xaxis=dict(
                title='Contribuição (%)',
                ticksuffix='%',
                gridcolor='rgba(31,58,95,0.1)',
                linecolor='rgba(31,58,95,0.2)',
                tickfont=dict(color='#1F3A5F'),
                title_font=dict(color='#1F3A5F'),
            ),
            yaxis=dict(
                autorange='reversed',
                gridcolor='rgba(31,58,95,0.1)',
                linecolor='rgba(31,58,95,0.2)',
                tickfont=dict(color='#1F3A5F'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#1F3A5F',
            showlegend=False,
            height=max(400, len(sorted_names) * 40),
            margin=dict(t=40, b=40, l=20, r=100),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    except Exception as e:
        st.warning(f'Não foi possível gerar a explicação SHAP: {e}')
