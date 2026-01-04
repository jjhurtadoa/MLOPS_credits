# Módulo `data`

> Carga y división de datos en train/test

## ⚠️ Evitar Data Leakage — por qué el split está en `src/data`

El módulo `data` realiza la partición train/test antes de cualquier transformación para prevenir data leakage.Hacer el split en una etapa separada garantiza que todas las transformaciones que «aprenden» del dato (imputadores, scalers, codificadores, selección de features) se ajusten únicamente sobre el conjunto de entrenamiento y luego se apliquen al test/producción.Buenas prácticas: partir los datos primero, ajustar preprocesadores solo con `train` (fit) y usar `random_state` fijo.Registrar la semilla y la versión/commit en la metadata para trazabilidad.

## 📂 Estructura
```
src/data/
├── load_data.py          # Funciones principales
└── run_load_data.py      # Script ejecutable
```

---

## 🎯 Propósito

- ✅ Cargar datos crudos desde CSV
- ✅ Validar estructura básica
- ✅ Dividir en train/test (80/20)
- ✅ Guardar splits en `data/splits/`

---

## 🚀 Uso Rápido

```bash
# Básico
python -m src.data.run_load_data

# Personalizado
python -m src.data.run_load_data \
    --input data/raw/HousingData.csv \
    --test-size 0.25 \
    --target MEDV
```

---

## 🔧 Funciones Principales

### `load_raw_data(data_path)`
Carga CSV y valida estructura.

**Retorna**:  `pd.DataFrame`

**Validaciones**:
- ✅ Archivo existe
- ✅ No está vacío
- ✅ Reporta valores faltantes

---

### `split_data(df, target_column, test_size=0.2)`
Divide en train/test.

**Retorna**: `(X_train, X_test, y_train, y_test)`

**Parámetros clave**:
- `test_size`: Proporción test (default:  0.2)
- `random_state`: Semilla (default: 42)

---

### `save_data(X_train, X_test, y_train, y_test)`
Guarda splits en CSV.

**Salida**:
```
data/splits/
├── train.csv
└── test.csv
```

---

## 📊 Pipeline

```
CSV Raw → Validación → Split (80/20) → Guardar
```

---


## ⚠️ Consideraciones

- **Reproducibilidad**: Usa `random_state=42`
- **No modifica**:  Solo carga y divide, no limpia
- **Siguiente paso**: `src.preprocess`

---

## 🔗 Ver también

- [Configuration](../configuration.md#data-config)
- [Pipeline completo](../pipelines.md)