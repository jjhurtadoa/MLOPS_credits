## MLOPS CREDITS

> Sistema MLOps para predicción de precio de vivienda

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)

## 🎯 Descripción

Proyecto MLOps que automatiza ingestión, preprocesado, entrenamiento y despliegue (MLflow + FastAPI + Docker) para predecir el precio de vivienda (Boston Housing).

## 🚀 Quick Start

```bash
# Clonar repositorio
git clone https://github.com/jjhurtadoa/MLOPS_credits.git
```

### Instalar dependencias

Breve explicación:
- `requirements.txt`: dependencias necesarias para el preprocesamiento y entrenamiento local (p. ej. pandas, scikit-learn, xgboost, lightgbm, catboost). Instalarlo si vas a ejecutar pipelines de datos o entrenar modelos.
- `requirements-api.txt`: dependencias mínimas para servir la API (p. ej. FastAPI, uvicorn, pydantic, joblib). Instalarlo si solo quieres ejecutar la API o construir la imagen runtime.

Comandos:

```bash
# Para preprocesamiento / entrenamiento local
pip install -r requirements.txt

# Para ejecutar solo la API (recomendado para producción/runtime)
pip install -r requirements-api.txt
```