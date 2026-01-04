# Módulo `preprocess`

> Preprocesamiento y feature engineering configurado mediante YAML

## 📂 Estructura
```
src/preprocess/
├── preprocessing.py         # Transformers de limpieza
├── build_features.py        # Feature engineering
└── run_preprocess.py        # Script ejecutable
```

---

## 🎯 Qué hace

Aplica transformaciones configuradas en **`preprocessing_config.yaml`**:

- ✅ Imputa valores faltantes (mean/median/moda)
- ✅ Trata outliers (log transform/IQR capping/winsorization)
- ✅ Elimina features correlacionadas (threshold)
- ✅ Escala datos (RobustScaler/StandardScaler)
- ✅ Crea features de interacción (producto/ratio/suma/diferencia)

---

## 🚀 Uso

```bash
# Básico
python -m src.preprocess.run_preprocess

# Config personalizada
python -m src.preprocess.run_preprocess --config custom_config.yaml

# Sin feature engineering
python -m src.preprocess.run_preprocess --no-feature-engineering
```

---

## ⚙️ Configuración

### **Archivo:** `src/configs/preprocessing_config.yaml`

**Generado desde:** `notebooks/eda.ipynb` (basado en análisis exploratorio)

**Estructura:**
```yaml
feature_processing:
  handling_na: # Estrategia de imputación (evita data leakage usando estadísticas de train)
    median: [CRIM, ZN, INDUS, AGE, LSTAT] # Variables con distribuciones sesgadas
    most_frequent: [CHAS]                 # Variables categóricas o binarias
    no_action: [NOX, RM, DIS, ...]        # Columnas sin valores nulos detectados
    
  handling_outliers: # Tratamiento de valores atípicos para estabilidad del modelo
    log_transform: [CRIM, ZN, LSTAT]      # Reduce el impacto de colas largas
    capping_iqr: [RM, B]                  # Limita valores fuera de [Q1-1.5IQR, Q3+1.5IQR]
    capping_winsor: [PTRATIO]             # Limita valores a percentiles específicos
    
  handling_correlated_features: # Reducción de dimensionalidad y multicolinealidad
    remove: [RAD, B]                      # Eliminadas por alta correlación (>0.85)
    
  scaler: RobustScaler # Escalado resistente a outliers (usando mediana y cuartiles)

feature_engineering: # Generación de nuevas variables predictoras
  domain_features: true # Activa cálculos específicos del dominio 
  interactions: []      # Cruce de variables (ej. X1 * X2)

```

> 📖 **Detalles completos:** Ver [Configuration](../configuration.md#preprocessing_configyaml)

---

## 📊 Clases Principales

| Clase | Propósito |
|-------|-----------|
| `MissingValueHandler` | Imputa valores faltantes según estrategia |
| `OutlierHandler` | Transforma/elimina outliers |
| `CorrelationReducer` | Elimina features correlacionadas |
| `DataPreprocessor` | Pipeline completo de limpieza |
| `InteractionFeatureCreator` | Crea features derivadas |
| `FeatureEngineer` | Pipeline de feature engineering |

**Todas son sklearn-compatible** (BaseEstimator, TransformerMixin)

---

## 📁 Salida

```
data/interim/
├── train_processed.csv           # Datos limpios y transformados
└── test_processed.csv

artifacts/preprocessors/
├── preprocessor.pkl              # Pipeline de limpieza
└── feature_engineer.pkl          # Pipeline de features
```

---

## 🔄 Modificar Config

### **Opción 1:  Editar YAML**
```bash
nano src/configs/preprocessing_config.yaml
python -m src.preprocess.run_preprocess
```

### **Opción 2: Regenerar desde EDA**
En `notebooks/eda.ipynb`, modificar `eda_config` y ejecutar celda de guardado.

> Ver [Configuration - Generación](../configuration.md#generación-desde-eda)

---

## ⚠️ Notas

- **Fit solo en train:** Transformers se ajustan SOLO en datos de entrenamiento
- **Persistencia:** `.pkl` se usan en producción con mismo preprocesamiento
- **Validación:** El código valida que features en config existan en datos

---

## 🔗 Ver también

- [Configuration](../configuration.md) - Guía completa de configs YAML
- [Data](./data.md) - Carga y división de datos
- [Train](./train.md) - Entrenamiento de modelos