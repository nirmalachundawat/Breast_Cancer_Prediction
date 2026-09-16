FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BREAST_CANCER_ARTIFACT_MOUNT_ROOT=/app/mlruns

WORKDIR /app

COPY requirements.docker.txt .

RUN pip install --no-cache-dir -r requirements.docker.txt

COPY app.py .
COPY src ./src
COPY monitoring ./monitoring
COPY templates ./templates
COPY data ./data

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/health')"

CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "app:app"]
