# Documentación MLOPS_credits

> Sistema MLOps para predicción de precios de viviendas usando Boston Housing Dataset

## 📋 Contenido

### 🏗️ Arquitectura

- [**Arquitectura del Sistema**](./architecture.md) - Diseño general del proyecto MLOps
- [**Pipelines End-to-End**](./pipelines.md) - Flujos de trabajo completos

### 📦 Módulos (`src/`)

- [**data**](./modules/data.md) - Carga y división de datos
- [**preprocess**](./modules/preprocess.md) - Preprocesamiento y feature engineering
- [**train**](./modules/train.md) - Entrenamiento de modelos con MLflow
- [**evaluate**](./modules/evaluate.md) - Evaluación y visualizaciones
- [**utils**](./modules/utils.md) - Utilidades compartidas (logging)

### 📊 Datos y Configuración

- [**Dataset**](./dataset.md) - Boston Housing Dataset
- [**Configuration**](./configuration.md) - Archivos YAML de configuración

### 🚀 Deployment

- [**MLflow**](./mlflow.md) - Tracking de experimentos
- [**API REST**](./api.md) - Servicio de predicción con FastAPI
- [**Docker**](./docker.md) - Containerización y deployment

### 📚 Dependencias

- [**Requirements**](./requirements.md) - Gestión de dependencias

---

## 🚀 Quick Start

```bash
# 1.Instalar dependencias
pip install -r requirements.txt

# 2.Cargar y dividir datos
python -m src.data.run_load_data

# 3.Preprocesar
python -m src.preprocess.run_preprocess

# 4.Entrenar modelos
python -m src.train.run_train

# 5.Evaluar
python -m src.evaluate.run_evaluate --model artifacts/models/best_model.pkl

# 6.Servir API (deployment)
pip install -r requirements.api.txt
uvicorn api.main:app --reload

# 7.Docker (alternativa)
docker build -t mlops-housing .
docker run -p 8000:8000 mlops-housing
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| **ML Framework** | Scikit-learn, XGBoost, LightGBM |
| **Tracking** | MLflow |
| **API** | FastAPI + Uvicorn |
| **Containerización** | Docker |
| **Feature Engineering** | Custom transformers (sklearn-compatible) |
| **Logging** | Python logging (centralizado) |
| **Configuración** | YAML |
| **Visualización** | Matplotlib, Seaborn |

---

## 📂 Estructura del Proyecto

```
MLOPS_credits/
├── data/
│   ├── raw/              # Datos originales
│   ├── splits/        # Train/test split
│   └── interim/          # Datos preprocesados
├── src/                  # Código fuente Python
│   ├── data/            # Carga de datos
│   ├── preprocess/      # Preprocesamiento
│   ├── train/           # Entrenamiento
│   ├── evaluate/        # Evaluación
│   ├── utils/           # Utilidades
│   └── configs/         # Configuraciones YAML
├── api/                  # API REST (FastAPI) 
├── artifacts/
│   ├── models/          # Modelos entrenados (.pkl)
│   ├── preprocessors/   # Transformers (.pkl)
│   ├── logs/            # Logs de ejecución
│   ├── mlruns/          # MLflow tracking
│   └── evaluation/      # Reportes y gráficos
├── notebooks/           # Análisis exploratorio (79.4%)
├── docs/                # Esta documentación
├── tests/               # Tests unitarios
├── Dockerfile           # Containerización 
├── requirements.txt     # Dependencias core 
├── requirements.api.txt # Dependencias API 
└── README.md
```

---

## 📊 Distribución de Código

Según composición del repositorio:

| Lenguaje | % | Propósito |
|----------|---|-----------|
| **Jupyter Notebook** | 79.4% | Análisis exploratorio, experimentación |
| **Python** | 20.4% | Código productivo (`src/`, `api/`) |
| **Dockerfile** | 0.2% | Containerización |

---



## 📖 Convenciones

### Logs

- **Nivel**:  `INFO` por defecto
- **Ubicación**: `artifacts/logs/`
- **Formato**: `YYYY-MM-DD HH:MM:SS - module - LEVEL - message`

### Artifacts

- **Modelos**: `artifacts/models/*.pkl`
- **Preprocessors**: `artifacts/preprocessors/*.pkl`
- **Evaluación**: `artifacts/evaluation/*.{json,png}`

### Datos

- **Raw**:  Nunca modificar `data/raw/`
- **splited**: Train/test splits en `data/splits/`
- **Interim**: Features engineering en `data/interim/`

---

## 🔗 Links Útiles

- [MLflow UI](http://localhost:5000) - `mlflow ui --port 5000`
- [API Docs](http://localhost:8000/docs) - Swagger UI automática
- [API Redoc](http://localhost:8000/redoc) - Documentación alternativa

---

## 👥 Autor

[Juan José Hurtado](https://github.com/jjhurtadoa)

## 📄 Licencia

MIT License.
