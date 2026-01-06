# Módulo `evaluate`

> Evaluación y generación de reportes visuales a partir de un modelo entrenado

## 📂 Estructura
```
src/evaluate/
├── evaluate_model.py       # Clase ModelEvaluator (métricas, visualizaciones, reports)
├── visualizations.py       # Funciones de gráficos (residuals, predictions, importance...)
└── run_evaluate.py         # Script ejecutable para generar reportes desde CLI
```

---

## 🎯 Propósito

Proporcionar una evaluación reproducible y de alta calidad para un modelo ya seleccionado (por ejemplo desde MLflow). No reemplaza al tracking de MLflow, sino que complementa con:

- métricas detalladas (RMSE, MAE, R², MAPE, max/median error),
- visualizaciones publicables (PNG) y
- reportes JSON listos para archivar en `artifacts/evaluation/`.

---

## 🔀 Relación con MLflow

- MLflow: tracking y comparación rápida entre runs.
- `evaluate/`: análisis final y diagnóstico por segmentos, preparado para decisiones de deploy.

---

## 🚀 Uso (CLI)

```bash
# Evaluar un modelo y generar reporte (usa data/interim/test_processed.csv)
python -m src.evaluate.run_evaluate --model artifacts/models/best_model.pkl

# Incluir evaluación sobre train (opcional)
python -m src.evaluate.run_evaluate --model artifacts/models/best_model.pkl --include-train

# Personalizar data dir o output dir
python -m src.evaluate.run_evaluate --model artifacts/models/xgboost_model.pkl --data-dir data/interim --output-dir artifacts/evaluation/xgboost
```

---

## 🧭 Clase principal: ModelEvaluator (`src/evaluate/evaluate_model.py`)

Resumen de la API pública:

- __init__(model_path: str, model_name: Optional[str] = None)
  - Carga el modelo desde disco (`joblib.load`).
  - Inicializa `metrics_` y `predictions_`.

- calculate_metrics(y_true, y_pred, prefix='') -> dict
  - Calcula: rmse, mae, r2, mape (%), max_error, median_error.
  - Devuelve un diccionario con claves opcionalmente prefijadas (ej. `test_rmse`).

- evaluate(X_test, y_test, X_train=None, y_train=None) -> dict
  - Ejecuta predicción en test y opcionalmente en train.
  - Guarda predicciones en `self.predictions_` y métricas en `self.metrics_`.
  - Detecta overfitting simple: si `train_rmse - test_rmse < -1.0` emite un WARNING.

- generate_visualizations(output_dir, X_test=None)
  - Genera y guarda PNGs para residuals, predicted vs actual, distribución de errores y feature importance (si el modelo expone `feature_importances_`).

- generate_report(output_dir, X_test=None)
  - Guarda `self.metrics_` como JSON y llama a `generate_visualizations`.

---

## 📊 Funciones de visualización (`src/evaluate/visualizations.py`)

- plot_residuals(y_true, y_pred, title, save_path)
  - Residuals vs predicted + histograma de residuales.

- plot_predictions(y_true, y_pred, title, save_path)
  - Scatter predicted vs actual con línea perfecta y R² anotado.

- plot_feature_importance(model, feature_names, title, top_n, save_path)
  - Barra horizontal para `feature_importances_` (si está presente).

- plot_error_distribution(y_true, y_pred, title, save_path)
  - Histogramas de errores absolutos y porcentuales.

- plot_model_comparison(comparison_df, metric, save_path)
  - Visual comparativa entre modelos (colorea el "mejor").

---

## 📁 Salida esperada

Al ejecutar `run_evaluate.py` con `--output-dir artifacts/evaluation` se generan, por modelo:

- `{model_name}_metrics.json`  — JSON con todas las métricas calculadas
- `{model_name}_residuals_test.png` — residual plot
- `{model_name}_predictions_test.png` — predictions vs actual
- `{model_name}_errors_test.png` — distribución de errores
- `{model_name}_feature_importance.png` — (si aplica)

Los nombres exactos se construyen desde `ModelEvaluator.model_name`.

---

## ⚠️ Notas importantes (consistentes con el código)

- El evaluador carga el modelo con `joblib.load` — el archivo debe ser accesible y compatible.
- `feature_importances_` es requerida por `plot_feature_importance`; si no existe la función retorna `None` y no escribe archivo.
- La detección de overfitting es intencionalmente simple (umbral de 1.0 en RMSE diferencial). Ajústalo en `evaluate_model.py` si necesitas otro criterio.
- El módulo no modifica modelos ni registros de MLflow — solo lee modelos y datos.

---

## 🔗 Ver también

- [Train](./train.md) — cómo generar `artifacts/models/*`
- [Preprocess](./preprocess.md) — cómo generar `data/interim/*_processed.csv` usados por el evaluador
- [MLflow (tools)](../tools/mlflow.md) — localizar el modelo ganador en el tracking server

---