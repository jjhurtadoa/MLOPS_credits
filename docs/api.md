# API REST

> Servicio de predicción con FastAPI

## 📂 Estructura
```
api/
├── main.py          # Aplicación FastAPI con endpoints
├── config.py        # Configuración (Pydantic Settings)
├── schemas.py       # Modelos Pydantic (request/response)
├── utils.py         # Loaders (modelo y preprocessors)
└── __init__.py
```

---

## 🎯 Propósito

Servir el modelo entrenado como API REST: 

- ✅ Carga modelo y preprocessors **una sola vez** al inicio (Singleton pattern)
- ✅ Aplica preprocesamiento automático (data_preprocessor + feature_engineer)
- ✅ Validación de datos con Pydantic
- ✅ Documentación automática (Swagger UI + ReDoc)
- ✅ Health check endpoint
- ✅ CORS habilitado
- ✅ Logging configurado

---

## 🚀 Uso

### **Instalación**
```bash
pip install -r requirements-api.txt
# fastapi, uvicorn, pydantic, pydantic-settings, gunicorn
```

### **Iniciar servidor**

#### **Desarrollo (con reload)**
```bash
uvicorn api.main:app --reload --port 8000
```

#### **Producción (con Gunicorn)**
```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 📡 Endpoints

### **1. Root - Información de la API**
```bash
curl http://localhost:8000/

# Response:
{
  "name": "Boston Housing Price Prediction API",
  "version": "1.0.0",
  "description": "API para predecir precios de viviendas en Boston usando ML",
  "model":  "best_model",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict": "/predict",
    "batch_predict": "/predict/batch"
  }
}
```

---

### **2. Health Check**
```bash
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessors_loaded": true,
  "model_name": "best_model",
  "version": "1.0.0"
}
```

---

### **3. Predicción Individual**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS":  4.09,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B":  396.90,
    "LSTAT": 4.98
  }'

# Response:
{
  "prediction": 24.5,
  "model_name": "best_model",
  "input_features": 13,
  "engineered_features": 15  # Después de preprocessing
}
```

**Flujo interno:**
```
Input (13 features) 
  → data_preprocessor (limpieza, scaling, etc.) 
  → feature_engineer (crea features derivadas) 
  → Modelo (predicción)
```

---

### **4. Predicción Batch**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type:  application/json" \
  -d '{
    "instances": [
      {
        "CRIM": 0.00632,
        "ZN":  18.0,
        "INDUS": 2.31,
        "CHAS":  0,
        "NOX": 0.538,
        "RM": 6.575,
        "AGE": 65.2,
        "DIS": 4.09,
        "RAD": 1,
        "TAX": 296.0,
        "PTRATIO": 15.3,
        "B": 396.90,
        "LSTAT": 4.98
      },
      {
        "CRIM": 0.02731,
        "ZN": 0.0,
        ... 
      }
    ]
  }'

# Response:
{
  "predictions": [24.5, 21.3],
  "count": 2,
  "model_name": "best_model"
}
```

---

## 📊 Documentación Interactiva

FastAPI genera docs automáticamente: 

### **Swagger UI**
```
http://localhost:8000/docs
```
- ✅ Probar endpoints interactivamente
- ✅ Ver schemas de request/response
- ✅ Ejemplos precargados

### **ReDoc**
```
http://localhost:8000/redoc
```
- ✅ Documentación más limpia
- ✅ Mejor para leer/compartir

---

## 🔧 Componentes

### **1. Configuración (`config.py`)**
Usa `pydantic-settings` para configuración centralizada: 

```python
class Settings(BaseSettings):
    app_name: str = "Boston Housing Price Prediction API"
    app_version: str = "1.0.0"
    model_path: str = "artifacts/models/best_model.pkl"
    # ... 
    
    class Config:
        env_file = ".env"  # Soporte para variables de entorno
```

**Ventaja:** Configuración por `.env` en desarrollo, env vars en producción.

---

### **2. Schemas (`schemas.py`)**
Validación automática con Pydantic:

```python
class HousingFeatures(BaseModel):
    CRIM: float = Field(... , ge=0)  # >= 0
    ZN: float = Field(..., ge=0, le=100)  # 0-100
    RM: float = Field(..., ge=0, le=20)  # Validación de rango
    # ...
```

**Si envías datos inválidos:**
```bash
curl -X POST . ../predict -d '{"CRIM": -1, ... }'

# Response:
{
  "detail": [
    {
      "loc": ["body", "CRIM"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

### **3. Loaders (`utils.py`)**

#### **Singleton Pattern:**
```python
# Se cargan UNA SOLA VEZ al inicio
preprocessor_loader. load_preprocessors(...)
model_loader.load_model(...)

# Requests subsecuentes usan caché (rápido)
```

#### **Preprocesamiento automático:**
```python
# En main.py
X_raw = features.to_dataframe()  # 13 features originales
X_final = preprocessor_loader.preprocess(X_raw)  # Preprocessing completo
prediction = model_loader.predict(X_final)
```

---

### **4. Lifespan Context (`main.py`)**
Reemplaza `@app.on_event("startup")` (deprecated):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Cargar modelo y preprocessors
    preprocessor_loader.load_preprocessors(...)
    model_loader.load_model(...)
    yield
    # SHUTDOWN: Cleanup (si necesario)
```

**Ventaja:** Modelo cargado ANTES de aceptar requests.

---

## 📁 Artifacts Requeridos

```
artifacts/
├── models/
│   └── best_model.pkl                  # Modelo entrenado
└── preprocessors/
    ├── data_preprocessor.pkl           # Preprocesamiento
    └── feature_engineer.pkl            # Feature engineering
```

**Asegúrate de correr antes:**
```bash
python -m src.preprocess.run_preprocess
python -m src.train.run_train --optimize
```

---



## 🔗 Ver también

- [Train](./modules/train. md) - Entrenamiento del modelo
- [Preprocess](./modules/preprocess.md) - Transformers usados en la API
- [Docker](./docker. md) - Deployment con Docker