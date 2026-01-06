# Documentación MLOPS_credits

> Sistema MLOps para predicción de precios de viviendas usando Boston Housing Dataset

## 📋 Contenido

### 🏗️ Arquitectura

- [**Arquitectura del Sistema**](./architecture.md) - Diseño general del proyecto MLOps

### 📦 Módulos (`src/`)

- [**data**](./modules/data.md) - Carga y división de datos
- [**preprocess**](./modules/preprocess.md) - Preprocesamiento y feature engineering
- [**train**](./modules/train.md) - Entrenamiento de modelos con MLflow
- [**evaluate**](./modules/evaluate.md) - Evaluación y visualizaciones

### 🚀 Tools

- [**MLflow**](.tools/mlflow.md) - Tracking de experimentos: cómo configurar el tracking server, registrar runs y buscar modelos.
- [**Docker**](.tools/docker.md) - Containerización y deployment.
- [**Github_Actions**](.tools/github_actions.md) - Integración continua: pipelines para tests,  build de imagen y publicación de artefactos.

### 🚀 Monitoring

- [**Prometheus**](.monitoring/prometheus.md) - Recolección de métricas: exporters y métricas expuestas por la API y el servicio de entrenamiento.
- [**Grafana**](.monitoring/grafana.md) - Dashboards y alertas: ejemplos de paneles para latencia, tasa de error y métricas de modelos.


### 🚀 Complementos

- [**API REST**](./api.md) - Servicio de predicción con FastAPI (endpoints, ejemplo de request/response y despliegue).
- [**Dataset**](./dataset.md) - Boston Housing Dataset, descripción breve.
- [**Configuration**](./configuration.md) - Archivos YAML de configuración y cómo personalizar experimentos.
- [**Tests**](./tests.md) - Estructura de tests, comandos pytest y criterios para tests de integración/funcionales.
- [**Model**](./model.md) - Convenciones de modelos y metadata: qué contiene `artifacts/models` y cómo versionar modelos.
- [**EDA**](./eda.md) - Notebook EDA y hallazgos clave del análisis exploratorio (insights útiles para features).
- [**ROADMAP**](./roadmap.md) - Mejoras futuras a corto, mediano y largo plazo.


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
├── notebooks/           # Análisis exploratorio 
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
