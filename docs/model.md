# Documentación del Modelo

## Mejor modelo para producción (documentación): **xgboost_optimized** (XGBRegressor)
- Metadata: `artifacts/models/xgboost_optimized_metadata.yaml`
- Test RMSE: 2.3455
- Test MAE: 1.8081
- Test R2: 0.9250
- Nota: mejora de RMSE ≈ 0.4144 respecto a la versión baseline de XGBoost.

> Nota rápida: si el pipeline guardó `best_model` sin optimizar (ej. se corrió sin `--optimize`), eso es un detalle de ejecución. Para la documentación usamos `xgboost_optimized`.

### Comparación compacta (test)
| Modelo | Test RMSE | Test MAE | Test R2 | CV RMSE (mean) |
|--------|----------:|---------:|--------:|---------------:|
| **xgboost_optimized** | **2.3455** | **1.8081** | **0.9250** | - |
| xgboost (baseline)     | 2.6896  | 1.9513  | 0.9014 | 3.9209 |
| catboost               | 3.0497  | 1.8682  | 0.8732 | 3.6347 |
| random_forest          | 3.1360  | 2.1479  | 0.8659 | 4.1115 |
| lightgbm               | 3.3945  | 2.2404  | 0.8429 | 3.7819 |
| ridge                  | 4.5973  | 2.9994  | 0.7118 | 4.7659 |
| mlp                    | 5.2786  | 3.1354  | 0.6200 | 6.5323 |
| elastic_net            | 5.3356  | 3.4929  | 0.6118 | 5.7577 |

Metadata completo: `artifacts/models/best_model_metadata.yaml` (baseline XGBoost) and `artifacts/models/xgboost_optimized_metadata.yaml` (optimizado).

---

Referencias rápidas:
- `artifacts/models/model_comparison.csv` — métricas por modelo
- `artifacts/models/xgboost_optimized_metadata.yaml` — parámetros optimizados y métricas finales
- MLflow UI: correr `mlflow ui --port 5000` para ver runs