FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training.py .

ENV DATA_PATH=/data/obesity.csv
ENV MODEL_DIR=/model

CMD ["python", "training.py"]
