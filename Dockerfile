# Slim runtime image for the API. Build: docker build -t jo-tax-law-rag .
# Pinned by digest (multi-arch index) so a rebuild cannot pull a different base;
# Dependabot (docker ecosystem) proposes digest bumps.
# python:3.12-slim = 3.12.15-slim-trixie, resolved 2026-10-25
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so code changes do not invalidate this layer. requirements.lock
# pins the whole closure with hashes (pip-compile --generate-hashes from requirements.txt);
# --no-deps: nothing is resolved at build time, and a changed file on PyPI fails the build.
COPY requirements.lock .
RUN pip install --require-hashes --no-deps -r requirements.lock

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
# uvicorn PID 1 so it receives SIGTERM and shuts down cleanly. --no-server-header
# drops "server: uvicorn"; the app sets the security headers itself.
# --no-access-log: the app logs every request itself (request_id, client_key,
# method, path, status, ms) and never an IP; uvicorn's line would add the peer IP.
CMD ["sh", "-c", "exec uvicorn app.main:create_app_from_env --factory --host 0.0.0.0 --port ${PORT} --no-server-header --no-access-log"]
