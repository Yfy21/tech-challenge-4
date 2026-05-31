import os
import pandas as pd
import pickle

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

pd.set_option('display.max_columns', None)

# Importação dos dados
DATA_PATH = os.environ.get('DATA_PATH', '../data/obesity.csv')
df = pd.read_csv(DATA_PATH)
df.columns = [col.lower() for col in df.columns]

# Ajuste de valores e tipos de dados
bool_cols = ['family_history', 'favc', 'smoke', 'scc']
numeric_cat_cols = ['fcvc', 'ncp', 'ch2o', 'faf', 'tue']
text_cat_cols = ['caec', 'calc', 'mtrans', 'obesity']

df['gender'] = df['gender'].replace({'Male': False, 'Female': True}).astype(bool)
df = df.drop(columns=['height', 'weight'])
df['age'] = df['age'].round().astype('int')
df[bool_cols] = df[bool_cols].replace({'yes': True, 'no': False}).astype(bool)
df[numeric_cat_cols] = df[numeric_cat_cols].round().astype('int')

df[['caec', 'calc']] = df[['caec', 'calc']].replace('no', 'No')
df[text_cat_cols] = df[text_cat_cols].astype('category')

# Treinando o modelo
X = df.drop(columns='obesity')
y = df['obesity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Preprocessamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age']),
        ('ordinal_cat_num', OrdinalEncoder(), numeric_cat_cols),
        ('ordinal_cat_text', OrdinalEncoder(
            categories=[
                ['No', 'Sometimes', 'Frequently', 'Always'],
                ['No', 'Sometimes', 'Frequently', 'Always'],
                ['Automobile', 'Motorbike', 'Public_Transportation', 'Walking', 'Bike']
            ]
        ), ['caec', 'calc', 'mtrans']),
    ],
    remainder='passthrough'
)

# Pipeline de avaliação de modelos
param_grid = [
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20],
    },
    {
        'classifier': [HistGradientBoostingClassifier(random_state=42)],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
    },
    {
        'classifier': [LogisticRegression(random_state=42, max_iter=1000)],
        'classifier__C': [0.1, 1.0, 10.0],
    },
]

eval_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
eval_pipeline = GridSearchCV(eval_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
print("Training model...")
eval_pipeline.fit(X_train, y_train)
best_clf = eval_pipeline.best_estimator_.named_steps['classifier']
print(f"Best classifier: {best_clf.__class__.__name__}")
print(f"Best CV score: {eval_pipeline.best_score_:.4f}")
print(f"Test accuracy: {eval_pipeline.score(X_test, y_test):.4f}")
print("")

# Reduzindo as features do melhor modelo para ver se sua performance melhora
print("Selecting best features...")
feature_selection_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectFromModel(clone(best_clf), threshold='mean')),
    ('classifier', clone(best_clf)),
])
print("Training model with selected features...")
cv_scores = cross_val_score(feature_selection_pipeline, X_train, y_train, cv=5, scoring='accuracy')
feature_selection_pipeline.fit(X_train, y_train)

print(f"Features kept: {feature_selection_pipeline.named_steps['selector'].get_support().sum()}")
print(f"CV score (with feature selection): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Test accuracy: {feature_selection_pipeline.score(X_test, y_test):.4f}")
print("")

# Selecionando o melhor modelo (com ou sem feature selection)
print("Deciding which model to save...")
score_no_selection = eval_pipeline.score(X_test, y_test)
score_with_selection = feature_selection_pipeline.score(X_test, y_test)

if score_with_selection > score_no_selection:
    final_model = feature_selection_pipeline
    print(f"Saving model WITH feature selection (test accuracy: {score_with_selection:.4f})")
else:
    final_model = eval_pipeline.best_estimator_
    print(f"Saving model WITHOUT feature selection (test accuracy: {score_no_selection:.4f})")

# Saving the model
MODEL_DIR = os.environ.get('MODEL_DIR', '../model')
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as file:
    pickle.dump(final_model, file)

with open(os.path.join(MODEL_DIR, 'model_data.pkl'), 'wb') as file:
    pickle.dump({
        'X': X, 'y': y,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test
    }, file)
