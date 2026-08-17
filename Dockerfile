FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin shoppilot

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra retrieval --extra embedding

COPY app ./app
COPY data/offline_catalog ./data/offline_catalog

RUN mkdir -p /var/lib/shoppilot/retrieval \
    /var/lib/shoppilot/output \
    /var/lib/shoppilot/uploaded \
    && chown -R shoppilot:shoppilot /app /var/lib/shoppilot

ENV SHOPPILOT_DATASET_DIR=/app/data/offline_catalog \
    SHOPPILOT_RETRIEVAL_INDEX_DIR=/var/lib/shoppilot/retrieval \
    SHOPPILOT_OUTPUT_DIR=/var/lib/shoppilot/output \
    SHOPPILOT_UPLOAD_DIR=/var/lib/shoppilot/uploaded \
    SHOPPILOT_MEMORY_FILE=/var/lib/shoppilot/preferences.json \
    SHOPPILOT_CHECKPOINT_DB=/var/lib/shoppilot/checkpoints.sqlite3

USER shoppilot
EXPOSE 8000

CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
