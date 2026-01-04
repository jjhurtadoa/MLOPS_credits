# Arquitectura del Sistema

## 🏗️ Diagrama de Componentes

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Datos     │────▶│ Preprocessing│────▶│  Training   │
│   Raw       │     │   Pipeline   │     │   Pipeline  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   MLflow    │
                                          │  Tracking   │
                                          └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Cliente   │────▶│  FastAPI     │────▶│   Model     │
│             │     │  REST API    │     │  Registry   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Monitoring  │
                    │  Dashboard   │
                    └──────────────┘
```

## 🔄 Flujo de MLOps

### 1.**Data Ingestion**
- Source:  CSV/Database
- Validación de esquema
- Versionado de datos

### 2.**Training Pipeline**
- Feature engineering
- Model training
- Experiment tracking (MLflow)
- Model validation

### 3.**Model Serving**
- REST API (FastAPI)
- Docker containerization
- Load balancing

### 4.**Monitoring**
- Drift detection
- Performance metrics
- Alerting system