FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencias primero: capa cacheable.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# El CSV no se hornea en la imagen (licencia y tamaño). Móntalo en /app/data/raw.
VOLUME ["/app/data/raw", "/app/artifacts"]

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["streamlit", "run", "app.py", \
  "--server.port=8501", "--server.address=0.0.0.0"]
