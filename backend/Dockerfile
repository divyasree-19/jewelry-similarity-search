# Search backend. Build context is the project root (see docker-compose.yml), so the same image also runs the
# maintenance scripts and tests (the compose "tools" service). Catalogue images, indexes and feature stores are
# mounted at runtime, never baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ backend/
COPY scripts/ scripts/
COPY eval/ eval/
COPY config/ config/
COPY data/categories.txt data/catalogue.csv data/products.csv data/
RUN mkdir -p data/catalogue data/faiss_index data/features && chown -R app:app /app

USER app
WORKDIR /app/backend
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
