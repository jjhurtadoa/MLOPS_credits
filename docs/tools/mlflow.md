# MLflow

> Plataforma de tracking de experimentos ML

## 🎯 Propósito

Registra **automáticamente** durante entrenamiento: 
- Parámetros (hiperparámetros, configuración)
- Métricas (RMSE, MAE, R², MAPE)
- Artifacts (modelos, gráficos)
- Metadata (timestamp, duración, git commit)

---

## 🚀 Uso

### **Iniciar UI**
```bash
mlflow ui --backend-store-uri artifacts/mlruns --port 5000
# http://localhost:5000
```

### **Ver experimentos**
```
boston_housing_models
├── random_forest_baseline (RMSE: 4.5)
├── xgboost_baseline (RMSE: 3.8) ← Mejor
├── lightgbm_baseline (RMSE: 4.0)
└── xgboost_optimized (RMSE: 3.2) ← Mejor optimizado
```

### **Comparar runs**
1. Seleccionar múltiples runs
2. Click "Compare"
3. Ver tabla/gráficos de métricas

---

## 📊 Integración Automática

El módulo `train/` integra MLflow: 

```python
# En src/train/train_model.py
with mlflow.start_run(run_name='xgboost_baseline'):
    mlflow.log_params(params)           # Hiperparámetros
    model. fit(X_train, y_train)
    mlflow.log_metric('test_rmse', rmse) # Métricas
    mlflow.sklearn.log_model(model, 'model') # Modelo
```

**Sin escribir código extra**, cada entrenamiento registra: 
- ✅ Todos los hiperparámetros
- ✅ Métricas train/test
- ✅ Cross-validation scores
- ✅ Feature importance
- ✅ Modelo serializado

---

## 🔍 Features Clave

### **1. Comparación de modelos**
Ver qué combinación de modelo/parámetros da mejores resultados. 

### **2. Reproducibilidad**
Cada run guarda: 
- Versión de código (git commit)
- Parámetros exactos
- Random seed

### **3. Model Registry** (opcional)
```python
mlflow.register_model(f"runs:/{run_id}/model", "HousePriceModel")
# Versiona modelos:  v1, v2, v3... 
# Estados:  Staging, Production, Archived
```

---

## 📂 Estructura

```
artifacts/mlruns/
└── 0/  (experiment_id)
    └── abc123/  (run_id)
        ├── artifacts/model/  # Modelo . pkl
        ├── metrics/          # RMSE, MAE, R²
        ├── params/           # max_depth, learning_rate
        └── tags/             # Metadata
```

---

## 🔀 MLflow vs Evaluate

| Pregunta | Herramienta |
|----------|-------------|
| ¿XGBoost o RandomForest? | ✅ MLflow |
| ¿max_depth=6 o max_depth=8? | ✅ MLflow |
| ¿Cuál tuvo mejor RMSE? | ✅ MLflow |
| ¿Por qué XGBoost falló en casa #42? | ✅ evaluate/ |
| ¿Errores por rango de precio? | ✅ evaluate/ |
| ¿Reporte para PM? | ✅ evaluate/ |

**Se complementan:**
```
MLflow (tracking) → Selecciona XGBoost (RMSE: 3.2)
         ↓
evaluate/ (validación) → Analiza dónde/por qué falla
         ↓
Decisión de deploy
```

> 📖 Ver [Evaluate](../modules/evaluate.md) para análisis profundo

---

## 💼 Workflow Completo

```bash
# 1. Entrenar modelos (MLflow tracking automático)
python -m src.train.run_train

# 2. Comparar en MLflow UI
mlflow ui
# → XGBoost gana con RMSE=3.2

# 3. Optimizar ganador (tracking automático)
python -m src.train.run_train --optimize --n-trials 100
# → RMSE mejora a 3.2

# 4. Validar con evaluate/
python
```
```python
from src.evaluate import ModelEvaluator
evaluator = ModelEvaluator('artifacts/models/best_model.pkl')
evaluator.evaluate(X_test, y_test)
evaluator.generate_report('artifacts/evaluation/')
# → Descubre problema en casas >$500k

# 5. Decisión informada
# MLflow:  métricas globales ✅
# evaluate/: análisis por segmento ✅
# → Deploy 
```

---

## 📖 Recursos

- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

## 🔗 Ver también

- [Train](../modules/train.md) - Integración con MLflow
- [Evaluate](../modules/evaluate.md) - Análisis complementario