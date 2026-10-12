FROM python:3.12.10-slim AS runner

LABEL maintainer="Aibar Alauov <aibar.alauov@narxoz.com>"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==25.0.1 && \
    pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/results && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser scripts/ /app/scripts/

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import src.benchmark; print('healthy')" || exit 1

ENTRYPOINT ["python", "src/benchmark.py"]
CMD ["--config", "configs/config.yaml"]
