# Configuración

> Archivos YAML que controlan el comportamiento del sistema

## 📂 Ubicación
```
src/configs/
├── preprocessing_config.yaml    # Preprocesamiento y feature engineering
└── model_config.yaml            # Modelos y hiperparámetros
```

---

## 🎯 Propósito

Los archivos de configuración **separan lógica de datos**, permitiendo:
- ✅ Modificar comportamiento sin cambiar código
- ✅ Versionado de configuraciones
- ✅ Experimentación rápida
- ✅ Reproducibilidad

---

## 📄 1.`preprocessing_config.yaml`

### **Origen**

Este archivo se **genera automáticamente** desde el notebook `eda.ipynb` basándose en hallazgos del análisis exploratorio.

### **Generación desde EDA**

**En `notebooks/eda.ipynb`**: 

```python
import yaml

# Configuración basada en hallazgos de EDA
eda_config = {
    'missing_values': {
        'mean': ['CRIM', 'ZN'],           # Variables con distribución normal
        'median': ['AGE', 'LSTAT', 'RM'], # Variables sesgadas
        'most_frequent': [],
        'no_action': []
    },
    
    'outliers': {
        'log_transform': ['CRIM', 'B'],   # Outliers extremos → Log transform
        'iqr_capping': ['RM'],            # Outliers moderados → IQR capping
        'winsorization': {
            'columns': ['LSTAT'],
            'limits': [0.05, 0.05]        # Winsorización 5% superior/inferior
        },
        'no_action': []
    },
    
    'correlation': {
        'threshold':  0.9,                 # TAX e INDUS correlacionan 0.92
        'method': 'pearson'
    },
    
    'scaling': {
        'method': 'robust'                # RobustScaler para datos con outliers
    },
    
    'feature_engineering': {
        'interactions': [
            {
                'type': 'product',
                'features': ['RM', 'LSTAT'],  # Interacción importante
                'name': 'RM_x_LSTAT'
            },
            {
                'type':  'ratio',
                'features': ['LSTAT', 'RM'],
                'name':  'LSTAT_per_RM'
            }
        ]
    }
}

# Guardar en YAML
with open('../src/configs/preprocessing_config.yaml', 'w', encoding='utf-8') as fh:
    yaml.dump(
        eda_config, 
        fh,
        default_flow_style=False,  # Formato legible (no inline)
        sort_keys=False,            # Mantiene orden del diccionario
        indent=2,                   # Indentación de 2 espacios
        allow_unicode=True          # Soporte UTF-8
    )

print('✓ Guardado:  src/configs/preprocessing_config.yaml')
```

### **Estructura Completa**

```yaml
# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# ----------------------------------------------------------------------------
# 1.MISSING VALUES
# ----------------------------------------------------------------------------
missing_values: 
  mean:               # Imputar con media (para distribuciones normales)
    - CRIM
    - ZN
  
  median:            # Imputar con mediana (para distribuciones sesgadas)
    - AGE
    - LSTAT
    - RM
  
  most_frequent:     # Imputar con moda (para categóricas)
    []
  
  no_action:          # No imputar
    []

# ----------------------------------------------------------------------------
# 2.OUTLIERS
# ----------------------------------------------------------------------------
outliers:
  log_transform:     # Transformación logarítmica (outliers extremos)
    - CRIM
    - B
  
  iqr_capping:       # Capping por IQR (outliers moderados)
    - RM
  
  winsorization:     # Winsorización (reemplazar extremos por percentiles)
    columns:
      - LSTAT
    limits:  [0.05, 0.05]  # 5% inferior y superior
  
  no_action:         # No tratar outliers
    []

# ----------------------------------------------------------------------------
# 3.CORRELATION
# ----------------------------------------------------------------------------
correlation:
  threshold: 0.9     # Eliminar features con correlación > 0.9
  method: pearson    # Método:  pearson, spearman, kendall

# ----------------------------------------------------------------------------
# 4.SCALING
# ----------------------------------------------------------------------------
scaling:
  method: robust     # Opciones: robust, standard

# ----------------------------------------------------------------------------
# 5.FEATURE ENGINEERING
# ----------------------------------------------------------------------------
feature_engineering:
  interactions: 
    - type: product              # Multiplicación
      features: [RM, LSTAT]
      name: RM_x_LSTAT
    
    - type: ratio                # División
      features: [LSTAT, RM]
      name:  LSTAT_per_RM
    
    # Más interacciones...
    # - type: diff               # Resta
    # - type: sum                # Suma
```

### **Modificación**

#### Opción 1: Editar manualmente

```bash
nano src/configs/preprocessing_config.yaml
```

#### Opción 2: Regenerar desde EDA

```python
# En notebooks/eda.ipynb, modificar eda_config y ejecutar celda de guardado
```

#### Opción 3: Crear versiones alternativas

```bash
# Crear variante
cp src/configs/preprocessing_config.yaml src/configs/preprocessing_v2.yaml

# Usar variante
python -m src.preprocess.run_preprocess --config src/configs/preprocessing_v2.yaml
```

---

## 📄 2.`model_config.yaml`

### **Origen**

Este archivo se crea **manualmente** con las especificaciones de modelos y hiperparámetros.

### **Cómo Mejorarlo**

#### **Situación Actual** (Manual):

```yaml
# Creado manualmente
models:
  xgboost:
    module: 'xgboost'
    class: 'XGBRegressor'
    params:
      max_depth: 6
      learning_rate: 0.1
```



#### **Mejora 1: Soporte para tuning automático en el repo**

Incluir un optimizador de hiperparámetros real en `src/train/optimize_model.py` (uso de Optuna). Ese módulo realiza búsquedas, registra ejecuciones en MLflow y devuelve los mejores parámetros. Para incorporar los resultados a la configuración puedes copiar manualmente `best_params` a `src/configs/model_config.yaml` o usar un script que exporte los parámetros al archivo YAML.

#### **Mejora 2: Múltiples configs para experimentación**

```
src/configs/
├── preprocessing_config.yaml
├── model_config_baseline.yaml      # Modelos simples
├── model_config_tuned.yaml         # Modelos optimizados
└── model_config_ensemble.yaml      # Ensemble de modelos
```

**Uso**:

```bash
# Entrenar con baseline
python -m src.train.run_train --config src/configs/model_config_baseline.yaml

# Entrenar con tuned
python -m src.train.run_train --config src/configs/model_config_tuned.yaml
```

---



### 3.**Experimentación**

```bash
# Crear variantes para experimentos
cp preprocessing_config.yaml preprocessing_no_log.yaml

# Editar:  Quitar log_transform
# Ejecutar y comparar resultados en MLflow
```

---

## 🔧 Ejemplo Completo:  Workflow

### 1.EDA → Generar `preprocessing_config.yaml`

```python
# En notebooks/eda.ipynb
# ...análisis ...
# Generar config basado en hallazgos
with open('../src/configs/preprocessing_config.yaml', 'w') as f:
    yaml.dump(eda_config, f, default_flow_style=False)
```

### 2.Hyperparameter Tuning → Generar `model_config.yaml`

```python
# En notebooks/hyperparameter_tuning.ipynb (crear)
# ...grid search ...
# Guardar mejores parámetros en config
with open('../src/configs/model_config.yaml', 'w') as f:
    yaml.dump(model_config, f, default_flow_style=False)
```

### 3.Ejecutar pipelines con configs

```bash
python -m src.preprocess.run_preprocess   # Usa preprocessing_config.yaml
python -m src.train.run_train             # Usa model_config.yaml
```

## Estructura compacta de `src/configs/`

- `preprocessing_config.yaml`  -> reglas de imputación, outliers, scaling y feature_engineering (generado desde `notebooks/eda.ipynb`).
- `model_config.yaml`         -> definición de modelos, parámetros y settings de entrenamiento (mlflow, validación).
- Posibles variantes: `model_config_baseline.yaml`, `model_config_tuned.yaml`, `preprocessing_v2.yaml`.

Nota: el `preprocessing_config.yaml` que se usa actualmente se deriva del notebook de EDA; puedes editarlo manualmente o regenerarlo desde el notebook para mantener consistencia con los hallazgos.

### Formato (resumen rápido)

- preprocessing_config.yaml (keys principales):
  - missing_values: {mean, median, most_frequent, no_action}
  - outliers: {log_transform, iqr_capping, winsorization}
  - correlation: {threshold, method}
  - scaling: {method}
  - feature_engineering: {interactions: [{type, features, name}]}

- model_config.yaml (keys principales):
  - training: {mlflow:{tracking_uri, experiment_name}, validation:{cv_folds, random_state}}
  - models: {<model_name>: {module, class, params}}
  - selection: {metric, mode}

## 🔗 Ver también

- [Preprocess Module](./modules/preprocess.md) - Uso de `preprocessing_config.yaml`
- [Train Module](./modules/train.md) - Uso de `model_config.yaml`
- [Notebooks](./notebooks.md) - Generación de configs desde EDA

---

