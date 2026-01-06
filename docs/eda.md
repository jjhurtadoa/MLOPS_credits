# EDA — Flujo compacto

Propósito: explorar datos crudos, detectar problemas (NA, outliers, correlaciones) y derivar decisiones que alimenten `preprocessing_config.yaml`.

Flujo resumido:
1. Cargar datos
   - `df = pd.read_csv('data/raw/HousingData.csv')`
2. Revisión rápida
   - `df.info()`, `df.describe()`, `df.isna().sum()`
3. Distribuciones y outliers
   - Histogramas / boxplots por variable (seaborn/matplotlib)
   - Decidir transformaciones (log, winsorize)
4. Correlaciones
   - Matriz de correlación, heatmap; eliminar features con corr > threshold
5. Missing values
   - Analizar patrón de NA y decidir imputación por columna (media/mediana/moda)
6. Feature engineering
   - Probar interacciones simples (producto, ratio), polinomios; evaluar gain en validación
7. Guardar artefactos
   - Exportar reportes (figuras) en `artifacts/` y escribir `src/configs/preprocessing_config.yaml` con las decisiones tomadas

Herramientas comunes: pandas, seaborn, matplotlib, numpy, scikit-learn (para transformaciones de prueba).

Notebook: `notebooks/eda.ipynb` contiene las celdas que generan los gráficos y la versión inicial de `preprocessing_config.yaml`.