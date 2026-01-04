# Módulo `evaluate`

> Análisis profundo de modelos complementario a MLflow

## 📂 Estructura
```
src/evaluate/
├── evaluate_model.py       # Clase ModelEvaluator
├── visualizations. py       # Funciones de gráficos
└── run_evaluate.py         # Script ejecutable

```

---

## 🎯 Propósito

Validación detallada del modelo **después** de seleccionarlo en MLflow: 

- ✅ Análisis de errores por segmentos de negocio
- ✅ Visualizaciones de alta calidad para presentaciones
- ✅ Reportes para stakeholders no-técnicos
- ✅ Validación pre-deploy

---

## 🔀 MLflow vs Evaluate

| Aspecto | MLflow | evaluate/ |
|---------|--------|-----------|
| **Fase** | Experimentación | Validación final |
| **Pregunta** | "¿Qué modelo ganó?" | "¿Por qué ganó?  ¿Dónde falla?" |
| **Usuarios** | Data Scientists | DS + Stakeholders |
| **Métricas** | Globales | Por segmentos |
| **Visualizaciones** | Básicas (UI) | Profesionales (PNG/HTML) |
| **Reportes** | No | Sí |

**Flujo típico:**
```
1. MLflow → XGBoost gana (RMSE:  3.2)
2. evaluate/ → Falla en casas >$500k (MAPE: 18.7%)
3. Decisión → Iterar antes de deploy
```

> 📖 Ver [MLflow](../tools/mlflow.md) para tracking de experimentos

---

## 🚀 Uso

```python
from src.evaluate import ModelEvaluator

# Cargar modelo ganador de MLflow
evaluator = ModelEvaluator('artifacts/models/best_model.pkl')

# Evaluar
metrics = evaluator.evaluate(X_test, y_test)

# Generar reportes
evaluator.generate_report('artifacts/evaluation/')
```

---

## 📊 Análisis Disponibles

### **1. Métricas detalladas**
```python
metrics = evaluator.calculate_metrics(y_test, y_pred)
# RMSE, MAE, R², MAPE, max_error, median_error
```

### **2. Visualizaciones**
```python
from src.evaluate.visualizations import (
    plot_residuals,           # Residual plot
    plot_predictions,         # Predicted vs Actual
    plot_feature_importance,  # Feature importance
    plot_error_distribution   # Error distribution
)
```

---

## 📁 Salida

```
artifacts/evaluation/
├── metrics_XGBoost. json            # Métricas JSON
├── residuals_XGBoost.png           # Residual plot
├── predictions_XGBoost.png         # Scatter plot
├── error_distribution_XGBoost. png  # Histograma de errores
└── feature_importance_XGBoost.png  # Bar plot
```

---

## 💡 Ejemplo de Valor Agregado

### **Situación:**
```
MLflow:   XGBoost RMSE=3.2 ✅
¿Deploy? 🤔
```

### **Análisis con evaluate/:**
```python
evaluator.analyze_by_price_range([200000, 500000])

# Output:
# - Casas <$200k:    MAPE = 8.5%  ✅
# - Casas $200-500k: MAPE = 10.2% ✅
# - Casas >$500k:  MAPE = 18.7% ⚠️ PROBLEMA

worst = evaluator.get_worst_predictions(n=5)
# Casa #42: Pred=$800k, Real=$400k (Error: 100%)
```

### **Decisión:**
```
NO deploy. 
Razón: Modelo falla en casas caras. 
Acción: Más datos de casas >$500k. 
```

**Sin evaluate/:** Habrías desplegado modelo con fallas graves 🚨

---

## ⚠️ Cuándo Usar

### **Usa evaluate/ cuando:**
- ✅ Tienes candidato para deploy
- ✅ Necesitas análisis por segmentos
- ✅ Presentarás resultados a stakeholders

### **Usa MLflow cuando:**
- ✅ Experimentas con múltiples modelos
- ✅ Comparas hiperparámetros
- ✅ Iteración rápida

---

## 🔗 Ver también

- [MLflow](../tools/mlflow.md) - Tracking de experimentos
- [Train](./train.md) - Entrenamiento de modelos