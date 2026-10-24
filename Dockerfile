# Slim runtime image for the API. Build: docker build -t jo-tax-law-rag .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so code changes do not invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app needs at runtime (see .dockerignore).
COPY app ./app
COPY migrations ./migrations
COPY data/clean ./data/clean
COPY data/eval ./data/eval

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# stdlib only: the slim image has no curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# Render (and docker-compose) pass PORT. "sh -c" expands ${PORT}; "exec" makes
# uvicorn PID 1 so it receives SIGTERM and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn app.main:create_app_from_env --factory --host 0.0.0.0 --port ${PORT}"]
