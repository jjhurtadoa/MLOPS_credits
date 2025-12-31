# ============================================================================
# Boston Housing MLOps - Production Dockerfile
# ============================================================================

FROM python:3.9-slim

# Metadata
LABEL maintainer="jjhurtadoa"
LABEL description="Boston Housing Price Prediction API"
LABEL version="1.0.0"

# Configurar entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Crear usuario no-root (seguridad)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (solo las necesarias)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copiar requirements PRIMERO (aprovechar cache de Docker)
COPY --chown=appuser:appuser requirements.txt requirements-api.txt ./

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements. txt && \
    pip install --no-cache-dir -r requirements-api.txt && \
    pip install --no-cache-dir gunicorn

# Copiar código fuente
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/

# Copiar artifacts (modelo + preprocessors)
COPY --chown=appuser:appuser artifacts/models/best_model.pkl ./artifacts/models/best_model.pkl
COPY --chown=appuser:appuser artifacts/preprocessors/ ./artifacts/preprocessors/

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto (producción con Gunicorn)
CMD ["gunicorn", "api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]