FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/model
ENV DATA_PATH=/data/obesity.csv
ENV API_URL=http://api:5000

EXPOSE 8501

CMD ["streamlit", "run", "Início.py", "--server.address=0.0.0.0"]
