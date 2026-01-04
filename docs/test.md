# Pruebas (tests) — guía específica del repo

Resumen rápido
- La carpeta `tests/` contiene pruebas para las distintas capas del proyecto:
  - `tests/test_data.py` — carga/división/IO de datos
  - `tests/test_preprocessing.py` — transformaciones y pipeline de preprocesamiento
  - `tests/test_training.py` — entrenamiento y persistencia de modelos
  - `tests/test_evaluate.py` — evaluación y métricas

Marcadores y fixtures importantes
- Marcadores usados en los tests:
  - `@pytest.mark.slow` — pruebas que tardan (marcarlas en CI como opcionales)
  - `@pytest.mark.integration` — tests de integración que requieren datos reales
- Fixture global de logging:
  - `configure_test_logging` (autouse, scope=session) configura logging para toda la sesión de pytest.
- Fixtures recurrentes:
  - `sample_boston_data`, `boston_like_data` — datos sintéticos para unit tests
  - `temp_data_dir`, `tmp_path` — directorios temporales para I/O
  - `real_boston_data` — carga datos reales si existen (usa `pytest.skip` si falta)

Cómo ejecutar (ejemplos prácticos)
- Ejecutar toda la suite (local):
  - `pytest -q`
- Ejecutar solo los tests de `data`:
  - `pytest -q tests/test_data.py`
- Ejecutar un test concreto (ejemplo `load_split_data` roundtrip):
  - `pytest -q tests/test_data.py::test_load_split_data_roundtrip`
- Ejecutar preprocessing tests:
  - `pytest -q tests/test_preprocessing.py`
- Ejecutar tests lentos (incluye slow):
  - `pytest -q -m slow`
- Ejecutar solo integración (si tienes datos reales):
  - `pytest -q -m integration`
- Filtrar por nombre o palabra clave:
  - `pytest -q -k "split and real"`
- Parar en el primer fallo (útil en local):
  - `pytest --maxfail=1 -q`

Ejecutar en Docker (si configuras un contenedor de tests)
- `docker compose run --rm api pytest -q`

Cobertura (opcional)
- `pytest --cov=src -q`

Buenas prácticas para los tests del repo
- Evitar depender de `data/raw` en unit tests; usar fixtures temporales (`temp_data_dir`) o datos sintéticos.
- Tests de integración deben estar marcados con `integration` y ejecutarse separadamente en CI.
- Mantener fixtures reutilizables en `tests/conftest.py` si se comparten entre archivos.
- Registrar run_id/paths en metadata cuando los tests escriben artefactos para ajudar debugging.

Errores comunes y soluciones rápidas
- FileNotFoundError en tests: asegúrate de ejecutar primero `python -m src.data.run_load_data` o usar fixtures que creen los ficheros temporales.
- Tests lentos: marcar con `@pytest.mark.slow` y excluirlos en CI rápido con `-m "not slow"`.

Si quieres que genere un `tests/conftest.py` con fixtures comunes o que añada instrucciones para ejecutar tests en CI (GitHub Actions), lo creo ahora.