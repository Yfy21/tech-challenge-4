import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] * {
    color: #F5F7FA !important;
}
[data-testid="stMain"] div[data-baseweb="select"] > div {
    background-color: #F5F7FA !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #F5F7FA !important;
    border-color: rgba(31,58,95,0.3) !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #1F3A5F !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] [data-baseweb="tag"] {
    background-color: #E0A458 !important;
}
/* hides slider thumb tooltip; class is stable per Streamlit version but may change on upgrade */
.e2ups025 {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
st.title('Dashboard Clínico — Fatores de Risco para Obesidade')

# Importação dos dados
DATA_PATH = os.environ.get('DATA_PATH', './data/obesity.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.lower() for col in df.columns]
    df['age'] = df['age'].round().astype(int)
    for col in ['fcvc', 'ncp', 'ch2o', 'faf', 'tue']:
        df[col] = df[col].round().astype(int)
    df[['caec', 'calc']] = df[['caec', 'calc']].replace('no', 'No')
    return df

df = load_data()

# Dicionários de tradução
obesity_pt = {
    'Insufficient_Weight': 'Abaixo do peso',
    'Normal_Weight': 'Peso normal',
    'Overweight_Level_I': 'Sobrepeso I',
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I': 'Obesidade I',
    'Obesity_Type_II': 'Obesidade II',
    'Obesity_Type_III': 'Obesidade III',
}
obesity_order = list(obesity_pt.values())
obesity_pt_rev = {v: k for k, v in obesity_pt.items()}

gender_pt = {'Male': 'Masculino', 'Female': 'Feminino'}
gender_pt_rev = {v: k for k, v in gender_pt.items()}

bool_pt = {'yes': 'Sim', 'no': 'Não'}
fcvc_pt = {1: 'Raramente', 2: 'Às vezes', 3: 'Sempre'}
ncp_pt = {1: '1 refeição', 2: '2 refeições', 3: '3 refeições', 4: '4 ou mais'}
ch2o_pt = {1: 'Menos de 1L', 2: 'De 1L a 2L', 3: 'Mais de 2L'}
faf_pt = {0: 'Nunca', 1: '1–2x/semana', 2: '3–4x/semana', 3: '5x+/semana'}
tue_pt = {0: 'Até 2h', 1: '3–5h', 2: 'Mais de 5h'}
caec_pt = {'No': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'}
calc_pt = {'No': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'}
mtrans_pt = {
    'Automobile': 'Carro',
    'Motorbike': 'Moto',
    'Bike': 'Bicicleta',
    'Public_Transportation': 'Transporte público',
    'Walking': 'A pé',
}

BRAND_COLORS = ['#1F3A5F', '#2F6F73', '#E0A458', '#DF745C', '#2C5185',
                '#3E9297', '#E8BB83', '#E79887', '#241F5F', '#2F4D73']
SHAP_COLORSCALE = [[0, '#1F3A5F'], [0.5, '#2F6F73'], [1, '#E0A458']]

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

MODEL_DIR = os.environ.get('MODEL_DIR', './model')

@st.cache_resource
def load_shap_data():
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    preprocessor = model.named_steps['preprocessor']
    selector = model.named_steps.get('selector')
    classifier = model.named_steps['classifier']
    X_t = preprocessor.transform(data['X_test'])
    feat_names = preprocessor.get_feature_names_out()
    if selector is not None:
        X_t = selector.transform(X_t)
        feat_names = feat_names[selector.get_support()]
    explainer = shap.TreeExplainer(classifier)
    sv = explainer.shap_values(X_t)
    if not isinstance(sv, list):
        sv = [sv[:, :, i] for i in range(sv.shape[2])]
    return X_t, list(feat_names), sv, list(classifier.classes_), data['X_test'], data['y_test']

def _beeswarm_offsets(shap_col, row_width=0.45):
    n = len(shap_col)
    if n == 0:
        return np.zeros(0)
    v_min, v_max = shap_col.min(), shap_col.max()
    if v_min == v_max:
        return np.zeros(n)
    n_bins = min(50, max(10, n // 5))
    edges = np.linspace(v_min - 1e-9, v_max + 1e-9, n_bins + 1)
    bin_ids = np.digitize(shap_col, edges) - 1
    offsets = np.zeros(n)
    for b in np.unique(bin_ids):
        idxs = np.where(bin_ids == b)[0]
        k = len(idxs)
        if k < 2:
            continue
        spread = row_width * min(1.0, k / 10)
        offsets[idxs] = np.linspace(-spread / 2, spread / 2, k)
    return offsets


BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#1F3A5F',
    xaxis=dict(
        gridcolor='rgba(31,58,95,0.1)',
        linecolor='rgba(31,58,95,0.2)',
        tickfont=dict(color='#1F3A5F'),
        title_font=dict(color='#1F3A5F'),
    ),
    yaxis=dict(
        gridcolor='rgba(31,58,95,0.1)',
        linecolor='rgba(31,58,95,0.2)',
        tickfont=dict(color='#1F3A5F'),
        title_font=dict(color='#1F3A5F'),
    ),
)

# Elementos da página - Filtros
st.sidebar.header('Filtros')

selected_obesity_pt = st.sidebar.multiselect('Nível de obesidade', options=obesity_order, default=obesity_order)
selected_obesity = [obesity_pt_rev[o] for o in selected_obesity_pt]

selected_gender_pt = st.sidebar.multiselect('Sexo biológico', options=['Masculino', 'Feminino'],
                                            default=['Masculino', 'Feminino'])
selected_gender = [gender_pt_rev[g] for g in selected_gender_pt]

age_min, age_max = int(df['age'].min()), int(df['age'].max())
age_range = st.sidebar.slider('Idade', age_min, age_max, (age_min, age_max))

# Aplicação dos filtros
filtered = df[
    df['obesity'].isin(selected_obesity) &
    df['gender'].isin(selected_gender) &
    df['age'].between(age_range[0], age_range[1])
].copy()

filtered['obesity_pt'] = filtered['obesity'].map(obesity_pt)

# Elementos da página - Cartões KPI
total = len(filtered)
overweight_levels = ['Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
pct_over   = len(filtered[filtered['obesity'].isin(overweight_levels)]) / total * 100 if total else 0
pct_normal = len(filtered[filtered['obesity'] == 'Normal_Weight']) / total * 100 if total else 0
pct_under  = len(filtered[filtered['obesity'] == 'Insufficient_Weight']) / total * 100 if total else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric('Total de pacientes', total)
k2.metric('Sobrepeso / Obesidade', f'{pct_over:.1f}%')
k3.metric('Peso normal', f'{pct_normal:.1f}%')
k4.metric('Abaixo do peso', f'{pct_under:.1f}%')

st.divider()

# Elementos da página - Gráficos de Distribuição
st.subheader('Distribuição dos Respondentes por Nível de Obesidade')
dist = filtered['obesity_pt'].value_counts().reindex(obesity_order).dropna().reset_index()
dist.columns = ['Nível', 'Contagem']
fig = px.bar(dist, x='Nível', y='Contagem', color='Nível',
             category_orders={'Nível': obesity_order},
             color_discrete_sequence=BRAND_COLORS)
fig.update_layout(showlegend=False, xaxis_title='Nível de Obesidade', yaxis_title='Respondentes', **BASE_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Elementos da Página - Fatores de Impacto
st.subheader('Estatísticas por Fator de Risco')

factor_options = {
    'Idade':                                      ('age', None, None),
    'Sexo biológico':                             ('gender', gender_pt, ['Masculino', 'Feminino']),
    'Histórico familiar de sobrepeso':            ('family_history', bool_pt, ['Não', 'Sim']),
    'Consome alimentos calóricos com frequência': ('favc', bool_pt, ['Não', 'Sim']),
    'Frequência de consumo de vegetais':          ('fcvc', fcvc_pt, ['Raramente', 'Às vezes', 'Sempre']),
    'Número de refeições principais':             ('ncp', ncp_pt, ['1 refeição', '2 refeições', '3 refeições', '4 ou mais']),
    'Come entre as refeições':                    ('caec', caec_pt, ['Não', 'Às vezes', 'Frequentemente', 'Sempre']),
    'Fumante':                                    ('smoke', bool_pt, ['Não', 'Sim']),
    'Consumo diário de água':                     ('ch2o', ch2o_pt, ['Menos de 1L', 'De 1L a 2L', 'Mais de 2L']),
    'Monitora calorias':                          ('scc', bool_pt, ['Não', 'Sim']),
    'Frequência de atividade física':             ('faf', faf_pt, ['Nunca', '1–2x/semana', '3–4x/semana', '5x+/semana']),
    'Tempo de tela diário':                       ('tue', tue_pt, ['Até 2h', '3–5h', 'Mais de 5h']),
    'Consumo de álcool':                          ('calc', calc_pt, ['Não', 'Às vezes', 'Frequentemente', 'Sempre']),
    'Meio de transporte':                         ('mtrans', mtrans_pt, ['Carro', 'Moto', 'Transporte público', 'A pé', 'Bicicleta']),
}

selected_factor = st.selectbox('Selecione um fator', list(factor_options.keys()))
col_name, col_map, col_order = factor_options[selected_factor]

plot_df = filtered[['obesity_pt', col_name]].copy()

if col_map:
    plot_df[col_name] = plot_df[col_name].map(col_map)

    factor_counts = plot_df[col_name].value_counts().reindex(col_order).dropna().reset_index()
    factor_counts.columns = [col_name, 'Contagem']
    top_fig = px.bar(factor_counts, y=col_name, x='Contagem',
                     color=col_name,
                     title='Número de Respondentes por Opção',
                     labels={col_name: selected_factor, 'Contagem': 'Nº de respondentes'},
                     category_orders={col_name: col_order},
                     color_discrete_sequence=BRAND_COLORS)
    top_fig.update_layout(showlegend=False, xaxis_title=None, **BASE_LAYOUT)
    st.plotly_chart(top_fig, use_container_width=True)

    grouped = plot_df.groupby(['obesity_pt', col_name]).size().reset_index(name='Contagem')
    grouped['Percentual'] = grouped.groupby(col_name)['Contagem'].transform(lambda x: round((x / x.sum() * 100), 2))
    bottom_fig = px.bar(grouped, x='obesity_pt', y='Percentual', color=col_name,
                 barmode='group',
                 title='Distribuição das Respostas em Níveis de Obesidade',
                 labels={'obesity_pt': 'Nível de obesidade', col_name: selected_factor, 'Percentual': 'Porcentagem'},
                 category_orders={'obesity_pt': obesity_order, col_name: col_order},
                 color_discrete_sequence=BRAND_COLORS)
    bottom_fig.update_layout(**BASE_LAYOUT)
    bottom_fig.update_yaxes(ticksuffix='%')
    st.plotly_chart(bottom_fig, use_container_width=True)
else:
    bottom_fig = px.box(plot_df, x='obesity_pt', y=col_name,
                 color='obesity_pt',
                 labels={'obesity_pt': 'Nível de obesidade', col_name: selected_factor},
                 category_orders={'obesity_pt': obesity_order},
                 color_discrete_sequence=BRAND_COLORS)
    for i, trace in enumerate(bottom_fig.data):
        trace.fillcolor = BRAND_COLORS[i % len(BRAND_COLORS)]
    bottom_fig.update_layout(showlegend=False, **BASE_LAYOUT)
    st.plotly_chart(bottom_fig, use_container_width=True)

st.divider()

st.subheader('Importância global das variáveis')
st.caption('Média do valor absoluto SHAP por variável, agregada entre todos os níveis de obesidade. Quanto maior a barra, maior o impacto médio daquela variável na classificação de novas amostras.')

try:
    X_shap_full, feat_names_raw, shap_vals_full, classes, X_test_raw, y_test_raw = load_shap_data()
    feat_names_pt = [FEATURE_NAME_PT.get(f, f) for f in feat_names_raw]
    _gender_bool = {'Male': False, 'Female': True}
    _gender_values = [_gender_bool[g] for g in selected_gender]

    _mask = (
        X_test_raw['gender'].isin(_gender_values) &
        X_test_raw['age'].between(age_range[0], age_range[1]) &
        y_test_raw.isin(selected_obesity)
    ).values
    if _mask.sum() == 0:
        st.warning('Nenhuma amostra do conjunto de teste corresponde aos filtros selecionados.')
    else:
        X_shap_filtered = X_shap_full[_mask]
        shap_vals_filtered = [sv[_mask] for sv in shap_vals_full]
        mean_abs_shap = np.round(np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_filtered], axis=0) * 100, 2)
        importance_df = pd.DataFrame({
            'Variável': feat_names_pt,
            'Importância': mean_abs_shap,
        }).sort_values('Importância').reset_index(drop=True)
        fig_imp = px.bar(importance_df, x='Importância', y='Variável', orientation='h',
                     labels={'Importância': 'Impacto médio (p.p.)', 'Variável': ''},
                     color_discrete_sequence=['#1F3A5F'])
        fig_imp.update_traces(hovertemplate='Impacto médio: %{x:.2f}%<extra></extra>')
        fig_imp.update_layout(showlegend=False, margin=dict(t=30, b=30), **BASE_LAYOUT)
        fig_imp.update_xaxes(ticksuffix='%')
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.subheader('Impacto das Variáveis na Classificação de Novas Amostras')
    st.caption('Para cada nível de obesidade, as variáveis são ordenadas pelo quanto influenciam positiva ou negativamente a classificação de uma nova amostra naquele nível. A cor indica o valor da variável: azul = baixo, verde = médio, amarelo = alto. Obs.: no caso do sexo, azul = homem e amarelo = mulher.')

    available_classes_pt = [c for c in obesity_order if obesity_pt_rev[c] in selected_obesity]
    selected_class_pt = st.selectbox(
        'Selecione o nível de obesidade', available_classes_pt, key='shap_class'
    )
    class_en = obesity_pt_rev[selected_class_pt]
    class_idx = classes.index(class_en)

    _bee_mask = (
        X_test_raw['gender'].isin(_gender_values) &
        X_test_raw['age'].between(age_range[0], age_range[1]) &
        (y_test_raw == class_en)
    ).values

    if _bee_mask.sum() == 0:
        st.warning('Nenhuma amostra do conjunto de teste corresponde aos filtros selecionados para esta classe.')
    else:
        X_shap_bee = X_shap_full[_bee_mask]
        sv_class = shap_vals_full[class_idx][_bee_mask]

        mean_abs = np.abs(sv_class).mean(axis=0)
        sorted_feat_idx = np.argsort(mean_abs)
        n_feat = len(sorted_feat_idx)

        fig_bee = go.Figure()
        for plot_pos, feat_idx in enumerate(sorted_feat_idx):
            fv = np.asarray(X_shap_bee[:, feat_idx]).ravel().astype(float)
            sv = np.round(sv_class[:, feat_idx] * 100, 2)
            fv_global = np.asarray(X_shap_full[:, feat_idx]).ravel().astype(float)
            fv_min, fv_max = fv_global.min(), fv_global.max()
            fv_norm = (fv - fv_min) / (fv_max - fv_min) if fv_max > fv_min else np.full(len(fv), 0.5)
            y_off = _beeswarm_offsets(sv)
            show_cb = (plot_pos == n_feat - 1)
            fig_bee.add_trace(go.Scatter(
                x=sv,
                y=plot_pos + y_off,
                mode='markers',
                marker=dict(
                    size=10,
                    color=fv_norm,
                    colorscale=SHAP_COLORSCALE,
                    cmin=0, cmax=1,
                    showscale=show_cb,
                    colorbar=dict(
                        title=dict(text='Valor da variável', font=dict(color='#1F3A5F', size=12)),
                        tickvals=[0.05, 0.95],
                        ticktext=['Baixo', 'Alto'],
                        tickfont=dict(color='#1F3A5F'),
                        thickness=15,
                        len=0.5,
                        x=1.02,
                    ) if show_cb else None,
                ),
                showlegend=False,
                hovertemplate='Impacto neste indivíduo: %{x:+.2f}%<extra></extra>',
            ))

        fig_bee.add_vline(x=0, line_color='rgba(31,58,95,0.25)', line_width=1)
        fig_bee.update_layout(
            xaxis=dict(
                title='Impacto na predição (p.p.)',
                ticksuffix='%',
                gridcolor='rgba(31,58,95,0.1)',
                linecolor='rgba(31,58,95,0.2)',
                tickfont=dict(color='#1F3A5F'),
                title_font=dict(color='#1F3A5F'),
            ),
            yaxis=dict(
                tickvals=list(range(n_feat)),
                ticktext=[feat_names_pt[i] for i in sorted_feat_idx],
                gridcolor='rgba(31,58,95,0.1)',
                linecolor='rgba(31,58,95,0.2)',
                tickfont=dict(color='#1F3A5F'),
                title_font=dict(color='#1F3A5F'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#1F3A5F',
            showlegend=False,
            height=max(600, n_feat * 60),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_bee, use_container_width=True)

except Exception as e:
    st.warning(f'Não foi possível carregar os dados SHAP: {e}')