# Pruebas (tests) — resumen 

Qué se ha cubierto con tests:
- Data I/O y carga/división (`tests/test_data.py`)
- Preprocessing y transformaciones (`tests/test_preprocessing.py`)
- Entrenamiento y persistencia de modelos (`tests/test_training.py`)
- Evaluación y métricas (`tests/test_evaluate.py`)
- API/integación mínima cuando aplica (`tests/test_api.py`)

Comandos rápidos:
- Ejecutar toda la suite (local):
  `pytest -q`
- Ejecutar un archivo de tests:
  `pytest -q tests/test_data.py`
- Ejecutar un test concreto:
  `pytest -q tests/test_data.py::test_load_split_data_roundtrip`
- Tests lentos:
  `pytest -q -m slow`
- Tests de integración:
  `pytest -q -m integration`
- Con cobertura:
  `pytest --cov=src -q`
- En Docker (si usas el servicio `api`):
  `docker compose run --rm api pytest -q`

Notas:
- Usa fixtures en `tests/conftest.py` para compartir datos y recursos.
- Marca tests costosos con `@pytest.mark.slow` para excluirlos en CI rápido.