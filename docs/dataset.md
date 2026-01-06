# Dataset (data/raw)

Archivo principal: `data/raw/HousingData.csv` (Boston Housing).

Breve: conjunto tabular con variables socioeconómicas y de vivienda; la columna objetivo es `MEDV` (precio medio de la vivienda, en miles de dólares en el dataset original).

Columnas:
- CRIM: tasa de criminalidad per cápita por ciudad.
- ZN: proporción de terrenos residenciales (zonas con lotes > 25,000 ft²).
- INDUS: proporción de superficie no comercial por ciudad.
- CHAS: variable indicadora de límite con el río Charles (1 si limita, 0 si no).
- NOX: concentración de óxidos de nitrógeno (parts per 10 million aproximadamente).
- RM: número medio de habitaciones por vivienda.
- AGE: proporción de unidades ocupadas construidas antes de 1940.
- DIS: distancias ponderadas a cinco centros de empleo en Boston.
- RAD: índice de accesibilidad a autopistas radiales.
- TAX: tasa impositiva por cada 10,000 USD de propiedad.
- PTRATIO: ratio alumno-profesor por ciudad.
- B: 1000(Bk - 0.63)^2 donde Bk es la proporción de población afroamericana por ciudad (estadística usada históricamente en el dataset).
- LSTAT: porcentaje de población con estatus socioeconómico bajo.
- MEDV: Median value of owner-occupied homes in $1000s (objetivo).

Notas:
- El CSV contiene valores vacíos marcados como `NA`; el pipeline de preprocessing aplica imputación/configuración según `src/configs/preprocessing_config.yaml`.
- Para un uso rápido, carga con pandas: `pd.read_csv('data/raw/HousingData.csv')`.