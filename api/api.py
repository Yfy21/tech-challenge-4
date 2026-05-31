import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = os.environ.get('MODEL_DIR', '../model')
with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

REQUIRED_FIELDS = [
      'gender', 'age', 'family_history', 'favc', 'fcvc', 'ncp',
      'caec', 'smoke', 'ch2o', 'scc', 'faf', 'tue', 'calc', 'mtrans'
]

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.get_json()
    missing = [field for field in REQUIRED_FIELDS if field not in user_data]
    if missing:
        return jsonify({'error': f'Favor preencher os campos: {missing}'}), 400

    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
