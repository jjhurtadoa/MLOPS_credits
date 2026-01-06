## MLOPS CREDITS

> Sistema MLOps para predicción de precio de vivienda

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
![Docker](https://img.shields.io/badge/docker-compose-blue)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

## 📖 Acerca del Proyecto

Este proyecto implementa un pipeline MLOps completo que automatiza:

- 📥 **Ingestión y carga de datos**
- 🔧 **Preprocesamiento y feature engineering**
- 🤖 **Entrenamiento y optimización de modelos** (con seguimiento en MLflow)
- 📊 **Evaluación y comparación de modelos**
- 🚀 **Despliegue de API** (FastAPI + Docker)
- 📈 **Monitoreo en producción** (Prometheus + Grafana)
- ✅ **Testing automatizado** (pytest)

## 📚 Documentación Completa

Para información detallada sobre arquitectura, diseño, decisiones técnicas y guías paso a paso, consulta:

**➡️ [Documentación completa en `/docs`](docs/)**

Incluye:

- 🏗️ Arquitectura del sistema
- 📊 Pipeline de datos y entrenamiento
- 🔧 Configuración de servicios (API, Prometheus, Grafana)
- 🧪 Estrategia de testing
- 📈 Monitoreo y métricas
- 🔄 CI/CD
- 🚀 Mejoras

---

## 🚀 Quick Start

```bash
# Clonar repositorio
git clone https://github.com/jjhurtadoa/MLOPS_credits.git
```

### Obteción modelo

Primera fase de entrenamiento de modelo

#### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

#### Crear entorno virtual

##### En Linux/macOS

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

##### En Windows

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate
```

#### Instalar dependencias

```bash
# Actualizar pip (recomendado)
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

#### Cargar datos (Más información en [`docs/modules/data.md`](docs/modules/data.md))

Correr pipeline para cargar los datos y dejarlos listos para procesar

```bash
python -m src.data.run_load_data
```

#### Procesamiento de  datos (Más información en [`docs/modules/preprocess.md`](docs/modules/preprocess.md))

Correr pipeline para preprocesar los datos y dejarlos listos para hacer predicción

```bash
python -m src.preprocess.run_preprocess
```

#### Entrenamiento  de  modelo (Más información en [`docs/modules/train.md`](docs/modules/train.md))

Correr pipeline para entrenar modelo

```bash
# Baseline
python -m src.train.run_train

# Baseline + Optimización  (Optimiza el mejor modelo)
python -m src.train.run_train --optimize --n-trials 100

# Solo algunos modelos
python -m src.train.run_train --models random_forest xgboost

# Config personalizada (revisar docs/config.md)
python -m src.train.run_train --config custom_config.yaml
```

#### Comparación modelos MLFLOW (Más información en [`docs/tools/mlflow.md`](docs/tools/mlflow.md))

Entrar a mlflow a revisar el experimento bostong_housing_models y comparar visualmente los modelos según las métricas que se quieran explorar

```bash
mlflow ui --backend-store-uri artifacts/mlruns
```

#### Evaluación mejor modelo (Más información en [`docs/modules/evaluate.md`](docs/modules/evaluate.md))

Explorar un poco más el comportamiento del mejor modelo

```bash
python -m src.evaluate.run_evaluate --model artifacts\models
```

Revisar artifacts/evaluation

### Levantamiento de servicios (FastAPI) Local (Más información en [`docs/api.md`](docs/api.md))

#### Instalar dependencias

```bash
# Actualizar pip (recomendado)
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements-api.txt
```

Disponer el modelo por medio de API (Localmente)

```bash
uvicorn api.main:app --reload --port 8000
```

#### Probar endpoints

Ingresar a [http://localhost:8000/docs](http://localhost:8000/docs)

### Levantamiento de servicios (Docker) Local (Más información en [`docs/tools/docker.md`](docs/tools/docker.md))

Comandos para crear y levantar los servicios con Docker.
Cuando tenemos el contenedor de docker tenemos el agregado de prometheus y grafana que nos ayudarán a monitorear el estado de la API

```bash
# Crear el contenedor
docker-compose build
```

```bash
# Levantar los servicios
docker-compose up -d
```

Abre en tu navegador:

<http://localhost:8000/docs> <br>
<http://localhost:9090> (Prometheus) <br>
<http://localhost:3000> (Grafana - user: admin, pass: admin)

```bash
# Apagar los servicios
docker-compose down
```

### Test (Más información en [`docs/tests.md`](docs/tests.md))

Test para cada módulo y para la API

```bash
# Test load data
pytest tests/test_data.py

# Test preprocess data
pytest tests/test_preprocessing.py

# Test training data
pytest tests/test_training.py

# Test evaluate data
pytest tests/test_evaluate.py

# Test api
pytest tests/test_api.py

```

### Subir cambios al repo

```bash
git add .
git commit -m "Descripción cambios"
git push origin main
```

#### Subir nuevo modelo al repo

##### Instalar Git LFS (si no lo tienes)

```bash
choco install git-lfs
```

##### Inicializar Git LFS en tu repo

```bash
git lfs install
```

##### Configurar tracking de archivos .pkl

```bash
git lfs track "*.pkl" 
git lfs track "artifacts/models/*.pkl"
```

##### Agregar .gitattributes (creado automáticamente)

```bash
git add .gitattributes
```

##### Agregar el modelo

```bash
git add artifacts/models/best_model.pkl -f
```

##### Commit y push

```bash
git commit -m "feat: add model with Git LFS" <br>
git push origin main
```

---

## Uso de herramientas AI

Usé herramientas de asistencia basadas en IA durante el desarrollo y la documentación:

- GitHub Copilot Chat (web) — orientación práctica sobre dockerización, monitorización y CI/CD; apoyo en redacción y propuestas.
- GitHub Copilot (extensión VS Code) — autocompletado, generación de snippets y plantillas para scripts/tests.
