FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

ENV MODEL_DIR=/model

CMD ["python", "api.py"]
