# Production Dockerfile - minimal runtime
FROM python:3.9-slim

LABEL maintainer="jjhurtadoa"
LABEL description="Boston Housing Price Prediction API"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Crear usuario no-root
RUN useradd -m -u 1000 appuser \
 && mkdir -p /app /app/artifacts/models /app/artifacts/preprocessors \
 && chown -R appuser:appuser /app

WORKDIR /app

# Dependencias sistema necesarias para runtime (evitar compilación pesada)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements de runtime (aprovecha cache)
COPY --chown=appuser:appuser requirements-api.txt ./requirements-api.txt

# Instalar dependencias Python de runtime
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements-api.txt \
 && pip install --no-cache-dir gunicorn

# Copiar código fuente
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser src/ ./src/

# Copiar modelo si quieres inmovilizarlo en la imagen (opcional)
# Si cambias frecuentemente el modelo, monta como volumen en docker-compose/dev.
COPY --chown=appuser:appuser artifacts/models/best_model.pkl ./artifacts/models/best_model.pkl
COPY --chown=appuser:appuser artifacts/preprocessors/ ./artifacts/preprocessors/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]