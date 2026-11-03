# ================= Stage 1: trainer =================
FROM python:3.12-slim AS trainer
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
COPY requirements-train.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
# Беремо твої готові артефакти, щоб не качати датасет і не чекати 30 хвилин на захисті
COPY artifacts/ ./artifacts/

# ================= Stage 2: runtime =================
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
# Встановлюємо ТІЛЬКИ легкі залежності для API
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
# Магія Multi-stage: копіюємо готову модель з етапу trainer
COPY --from=trainer /build/artifacts ./artifacts

# Конфігурація через змінні середовища
ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    PYTHONUNBUFFERED=1

# Безпека: створюємо користувача без прав root і перемикаємося на нього
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Health-check для перевірки стану контейнера
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]