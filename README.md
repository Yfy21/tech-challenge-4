# Preditor de Nível de Obesidade

Aplicação preditiva desenvolvida como Tech Challenge da Fase 4 do programa Pós-Tech em Data Analytics da FIAP. O sistema combina um modelo de machine learning com uma interface interativa para estimar o nível de obesidade de um indivíduo com base em seus hábitos de vida, e um painel analítico para explorar os dados e entender o comportamento do modelo.

## Páginas da aplicação

### Preditor
Formulário com 14 perguntas sobre características demográficas e hábitos do usuário. Ao submeter, o modelo classifica o resultado em uma das sete categorias — de *Abaixo do peso* a *Obesidade III* — e exibe um gráfico SHAP waterfall indicando quais fatores mais influenciaram aquela predição específica e em que direção.

### Dashboard
Painel analítico com filtros por nível de obesidade, sexo biológico e faixa etária. Inclui:
- Indicadores gerais da base de dados
- Distribuição dos respondentes por nível de obesidade
- Análise estatística por fator de risco
- Importância global das variáveis via SHAP
- Impacto por nível de obesidade via beeswarm SHAP

## Arquitetura

```
┌─────────────┐       HTTP POST /predict      ┌─────────────┐
│  Streamlit  │ ────────────────────────────► │  Flask API  │
│    (app)    │                               │    (api)    │
└─────────────┘                               └─────────────┘
       │                                             │
       └──────────────── model.pkl ─────────────────┘
```

- **Streamlit**: interface do usuário (preditor + dashboard)
- **Flask API**: serviço de predição que recebe os dados do formulário e retorna a classificação
- **Treinamento**: pipeline scikit-learn com GridSearchCV, feature selection e RandomForest; gera `model.pkl` e `model_data.pkl`

## Tecnologias

| Camada | Tecnologias |
|---|---|
| Interface | Streamlit, Plotly |
| API | Flask |
| Modelo | scikit-learn (RandomForestClassifier), SHAP |
| Dados | pandas, numpy |
| Infraestrutura | Docker, Docker Compose |

## Como executar localmente

Pré-requisito: [Docker](https://www.docker.com/) instalado.

```bash
git clone https://github.com/Yfy21/tech-challenge-4.git
cd tech-challenge-4
docker compose up
```

O comando acima treina o modelo, sobe a API e inicia a aplicação Streamlit. Acesse em **http://localhost:8501**.

> O serviço de treinamento gera o modelo na primeira execução. Nas execuções seguintes, o modelo já estará disponível no volume Docker.

## Estrutura do projeto

```
.
├── api/                  # Flask API de predição
│   ├── api.py
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   └── obesity.csv       # Dataset de obesidade
├── model/                # Artefatos gerados pelo treinamento
│   ├── model.pkl
│   └── model_data.pkl
├── streamlit/            # Aplicação Streamlit
│   ├── Início.py
│   ├── pages/
│   │   ├── 1 Preditor.py
│   │   └── 2 Dashboard.py
│   ├── Dockerfile
│   └── requirements.txt
├── training/             # Pipeline de treinamento
│   ├── training.py
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```
