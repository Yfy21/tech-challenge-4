FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

ENV MODEL_DIR=/model

EXPOSE 5000

CMD ["python", "api.py"]
