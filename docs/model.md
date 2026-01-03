# Documentación del Modelo

## 🤖 Modelo Seleccionado:  XGBoost

### Experimentos Realizados

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.78 | 0.76 | 0.72 | 0.74 |
| Random Forest | 0.83 | 0.81 | 0.79 | 0.80 |
| **XGBoost** | **0.87** | **0.85** | **0.84** | **0.84** |

### Hiperparámetros Finales

```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### Feature Importance

1. `credit_score` - 0.35
2. `income` - 0.25
3. `debt_to_income_ratio` - 0.18
4. `employment_years` - 0.12
5. `age` - 0.10

### Tracking con MLflow

```bash
mlflow ui --port 5000
```

Ver experimentos en: http://localhost:5000