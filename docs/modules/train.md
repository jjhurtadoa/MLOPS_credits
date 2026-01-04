# Módulo `train`

> Entrenamiento de modelos con MLflow tracking y optimización opcional

## 📂 Estructura

```
src/train/
├── train_model.py       # Clase ModelTrainer (baseline)
├── optimize_model.py    # Clase ModelOptimizer (Optuna)
└── run_train.py         # Script ejecutable
```

---

## 🎯 Qué hace

**Baseline (defaults):**

- Entrena múltiples modelos con parámetros por defecto
- Compara familias (árboles vs lineales vs redes neuronales)
- Selecciona el mejor según métrica (RMSE)
- Registra todo en MLflow

**Optimización (opcional):**

- Optimiza hiperparámetros del ganador con Optuna
- Mejora típica: ~10-20% en RMSE

---

## 🚀 Uso

```bash
# Baseline rápido (~5 min)
python -m src.train.run_train

# Baseline + Optimización (~60 min)
python -m src.train.run_train --optimize --n-trials 100

# Solo algunos modelos
python -m src.train.run_train --models xgboost lightgbm

# Config personalizada
python -m src.train.run_train --config custom_config.yaml
```

---

## ⚙️ Configuración

### **Archivo:** `src/configs/model_config.yaml`

**Filosofía:** Parámetros por **defecto de las librerías** (sin inventar valores).

**Estructura:**

```yaml
models:
  xgboost:
    module: 'xgboost'
    class:  'XGBRegressor'
    params:
      random_state: 42  # Solo para reproducibilidad
      n_jobs: -1        # Solo para velocidad
      # Resto:  DEFAULTS (max_depth=6, learning_rate=0.3, etc.)
  
  lightgbm:
    module: 'lightgbm'
    class: 'LGBMRegressor'
    params:
      random_state: 42
      verbose: -1

training:
  mlflow:
    tracking_uri:  'artifacts/mlruns'
    experiment_name:  'boston_housing_models'
  
validation:
  cv_folds: 5
  cv_scoring: neg_root_mean_squared_error
```

> 📖 **Detalles:** Ver [Configuration](../configuration.md#model_configyaml)

---

## 📊 Modelos Ejecutados

| Familia | Modelos |
|---------|---------|
| **Árboles** | RandomForest, XGBoost, LightGBM, CatBoost |
| **Lineales** | Ridge, ElasticNet, Lasso |
| **Redes Neuronales** | MLPRegressor |

**Import dinámico:** Carga clases automáticamente desde módulos.

---

## 📁 Salida

### **Sin `--optimize`:**

```
artifacts/models/
├── xgboost_model.pkl              # Todos los modelos baseline
├── lightgbm_model.pkl
├── ...
├── best_model_baseline.pkl        # Mejor con defaults
├── best_model.pkl                 # ✅ FINAL (baseline)
├── model_comparison_baseline.csv  # Comparación
└── best_model_metadata.yaml       # Metadata
```

### **Con `--optimize`:**

```
artifacts/models/
├── xgboost_model.pkl              # Baseline
├── best_model_baseline.pkl        # Mejor baseline (RMSE:  3.8)
├── xgboost_optimized.pkl          # Optimizado con Optuna (RMSE: 3.2)
├── best_model.pkl                 # ✅ FINAL (optimizado)
└── best_model_metadata.yaml       # Incluye mejora:  15.8%
```

---

## 🔍 Optimización con Optuna

### **¿Cuándo se activa?**

Solo con flag `--optimize`.

### **¿Qué hace?**

1.Identifica el mejor modelo del baseline (ej: XGBoost) 
2.Busca mejores hiperparámetros con Bayesian Optimization  
3.Entrena modelo final con parámetros óptimos  
4.Compara baseline vs optimizado  


---

## 📈 MLflow Integration

### **Ver experimentos:**

```bash
mlflow ui --backend-store-uri artifacts/mlruns
# http://localhost:5000
```

### **Métricas tracked:**

- RMSE, MAE, R², MAPE (train y test)
- Cross-validation scores
- Feature importance
- Hiperparámetros
- Tiempos de entrenamiento

---

## 🎯 Flujo Completo

```
1.Cargar model_config.yaml (solo random_state)
   ↓
2.Entrenar todos con DEFAULTS
   ↓
3.Comparar en MLflow → XGBoost gana (RMSE: 3.8)
   ↓
4.¿--optimize?
   ├─ NO → Guardar best_model.pkl (baseline)
   └─ SÍ → Optuna en XGBoost
           ↓
           Mejores params (max_depth=8, lr=0.05, ...)
           ↓
           Entrenar modelo final (RMSE: 3.2)
           ↓
           Guardar best_model.pkl (optimizado)
```

---

## 💡 Argumentos Útiles

| Argumento | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--optimize` | Activar optimización Optuna | `--optimize` |
| `--n-trials` | Número de trials Optuna | `--n-trials 50` |
| `--models` | Entrenar solo algunos | `--models xgboost lightgbm` |
| `--config` | Config personalizada | `--config custom.yaml` |

---

## ⚠️ Notas

- **Defaults primero:** Comparación justa entre familias sin bias
- **Optuna después:** Solo al ganador para maximizar mejora
- **MLflow siempre:** Tracking automático de todo
- **Reproducibilidad:** `random_state=42` en todos los modelos

---

## 🔗 Ver también

- [Configuration](../configuration.md#model_configyaml) - Configuración de modelos
- [Evaluate](./evaluate.md) - Evaluación de modelos
- [MLflow](../mlflow.md) - Tracking de experimentos
