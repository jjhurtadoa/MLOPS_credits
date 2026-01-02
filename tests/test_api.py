"""
Tests para la API de predicción
"""

import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path

from api.main import app
from api.utils import model_loader, preprocessor_loader  # ← IMPORTAR


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def setup_api():
    """
    Setup:  Cargar modelo y preprocessors ANTES de los tests
    
    Esto es necesario porque TestClient no ejecuta lifespan events automáticamente
    """
    # Verificar que existen los artifacts
    model_path = Path("artifacts/models/best_model.pkl")
    data_preprocessor_path = Path("artifacts/preprocessors/data_preprocessor.pkl")
    feature_engineer_path = Path("artifacts/preprocessors/feature_engineer.pkl")
    
    if not model_path.exists():
        pytest.skip(f"Modelo no encontrado:  {model_path}")
    
    if not data_preprocessor_path.exists():
        pytest.skip(f"Data preprocessor no encontrado: {data_preprocessor_path}")
    
    if not feature_engineer_path.exists():
        pytest.skip(f"Feature engineer no encontrado: {feature_engineer_path}")
    
    # Cargar preprocessors
    preprocessor_loader.load_preprocessors(
        str(data_preprocessor_path),
        str(feature_engineer_path)
    )
    
    # Cargar modelo
    model_loader.load_model(str(model_path))
    
    yield
    
    # Teardown (si fuera necesario)
    pass


@pytest.fixture
def client(setup_api):  # ← Depende de setup_api
    """Cliente de test para la API"""
    return TestClient(app)


@pytest.fixture
def valid_input():
    """Input válido para predicción"""
    return {
        "CRIM": 0.00632,
        "ZN": 18.0,
        "INDUS": 2.31,
        "CHAS": 0,
        "NOX": 0.538,
        "RM": 6.575,
        "AGE": 65.2,
        "DIS": 4.0900,
        "RAD": 1,
        "TAX": 296.0,
        "PTRATIO": 15.3,
        "B": 396.90,
        "LSTAT":  4.98
    }


# ============================================================================
# RESTO DE TESTS (sin cambios)
# ============================================================================

def test_root_endpoint(client):
    """Test:  GET / retorna información de la API"""
    response = client.get("/")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_check_endpoint(client):
    """Test: GET /health retorna estado del servicio"""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "preprocessors_loaded" in data
    assert "model_name" in data
    assert "version" in data
    
    # Debe estar healthy
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessors_loaded"] is True


def test_predict_endpoint_success(client, valid_input):
    """Test: POST /predict con input válido retorna predicción"""
    response = client.post("/predict", json=valid_input)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "model_name" in data
    assert "model_version" in data
    
    # Precio debe ser razonable (Boston housing:  5-50k USD)
    assert 5.0 <= data["predicted_price"] <= 50.0
    assert isinstance(data["predicted_price"], float)


def test_predict_endpoint_validation_error_negative_crim(client, valid_input):
    """Test: POST /predict con CRIM negativo retorna error de validación"""
    invalid_input = valid_input.copy()
    invalid_input["CRIM"] = -1.0
    
    response = client.post("/predict", json=invalid_input)
    
    assert response.status_code == 422


def test_predict_endpoint_validation_error_zn_out_of_range(client, valid_input):
    """Test: POST /predict con ZN > 100 retorna error"""
    invalid_input = valid_input.copy()
    invalid_input["ZN"] = 150.0
    
    response = client.post("/predict", json=invalid_input)
    
    assert response.status_code == 422


def test_predict_endpoint_validation_error_chas_invalid(client, valid_input):
    """Test: POST /predict con CHAS inválido retorna error"""
    invalid_input = valid_input.copy()
    invalid_input["CHAS"] = 2
    
    response = client.post("/predict", json=invalid_input)
    
    assert response.status_code == 422


def test_predict_endpoint_validation_error_missing_field(client, valid_input):
    """Test: POST /predict con campo faltante retorna error"""
    invalid_input = valid_input.copy()
    del invalid_input["RM"]
    
    response = client.post("/predict", json=invalid_input)
    
    assert response.status_code == 422


def test_predict_endpoint_validation_error_extra_field(client, valid_input):
    """Test: POST /predict con campo extra funciona (Pydantic lo ignora)"""
    input_with_extra = valid_input.copy()
    input_with_extra["EXTRA_FIELD"] = 999
    
    response = client.post("/predict", json=input_with_extra)
    
    # Debe funcionar (Pydantic ignora extras)
    assert response.status_code == 200


def test_predict_endpoint_different_inputs(client):
    """Test: POST /predict con diferentes inputs válidos"""
    
    # Casa de alta calidad
    high_quality = {
        "CRIM": 0.01, "ZN": 50.0, "INDUS":  2.0, "CHAS":  1,
        "NOX": 0.4, "RM": 8.0, "AGE": 10.0, "DIS": 8.0,
        "RAD": 1, "TAX": 250.0, "PTRATIO": 12.0, "B": 396.90, "LSTAT": 2.0
    }
    
    response1 = client.post("/predict", json=high_quality)
    assert response1.status_code == 200
    price1 = response1.json()["predicted_price"]
    
    # Casa de baja calidad
    low_quality = {
        "CRIM": 15.0, "ZN": 0.0, "INDUS":  20.0, "CHAS":  0,
        "NOX": 0.8, "RM": 4.5, "AGE": 95.0, "DIS": 1.5,
        "RAD": 24, "TAX": 666.0, "PTRATIO": 22.0, "B": 200.0, "LSTAT":  30.0
    }
    
    response2 = client.post("/predict", json=low_quality)
    assert response2.status_code == 200
    price2 = response2.json()["predicted_price"]
    
    # Casa de alta calidad debe costar más
    assert price1 > price2


def test_predict_batch_endpoint_success(client, valid_input):
    """Test: POST /predict/batch con inputs válidos retorna predicciones"""
    batch_input = {
        "instances": [valid_input, valid_input.copy()]
    }
    
    response = client.post("/predict/batch", json=batch_input)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert "model_name" in data
    
    assert len(data["predictions"]) == 2
    assert data["count"] == 2


def test_predict_batch_endpoint_single_instance(client, valid_input):
    """Test: POST /predict/batch con una sola instancia"""
    batch_input = {"instances": [valid_input]}
    
    response = client.post("/predict/batch", json=batch_input)
    
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predict_batch_endpoint_multiple_instances(client, valid_input):
    """Test: POST /predict/batch con múltiples instancias"""
    batch_input = {
        "instances": [valid_input.copy() for _ in range(5)]
    }
    
    response = client.post("/predict/batch", json=batch_input)
    
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 5


def test_predict_batch_endpoint_validation_error(client, valid_input):
    """Test: POST /predict/batch con una instancia inválida retorna error"""
    invalid_instance = valid_input.copy()
    invalid_instance["CRIM"] = -5.0
    
    batch_input = {
        "instances": [valid_input, invalid_instance]
    }
    
    response = client.post("/predict/batch", json=batch_input)
    
    assert response.status_code == 422


def test_predict_batch_endpoint_empty_list(client):
    """Test: POST /predict/batch con lista vacía retorna error"""
    batch_input = {"instances": []}
    
    response = client.post("/predict/batch", json=batch_input)
    
    assert response.status_code == 422


def test_predict_vs_batch_consistency(client, valid_input):
    """Test: /predict y /predict/batch retornan mismo resultado"""
    
    # Predicción individual
    response_single = client.post("/predict", json=valid_input)
    price_single = response_single.json()["predicted_price"]
    
    # Predicción batch
    batch_input = {"instances": [valid_input]}
    response_batch = client.post("/predict/batch", json=batch_input)
    price_batch = response_batch.json()["predictions"][0]
    
    # Deben ser iguales
    assert abs(price_single - price_batch) < 0.0001


def test_predict_endpoint_invalid_json(client):
    """Test: POST /predict con JSON inválido retorna error"""
    response = client.post(
        "/predict",
        data="invalid json{",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422


def test_predict_endpoint_wrong_content_type(client, valid_input):
    """Test: POST /predict con Content-Type incorrecto"""
    response = client.post(
        "/predict",
        data=json.dumps(valid_input),
        headers={"Content-Type":  "text/plain"}
    )
    
    assert response.status_code in [422, 415]


def test_nonexistent_endpoint(client):
    """Test: GET a endpoint inexistente retorna 404"""
    response = client.get("/nonexistent")
    
    assert response.status_code == 404


def test_predict_response_time(client, valid_input):
    """Test: Tiempo de respuesta de /predict es razonable"""
    import time
    
    start = time.time()
    response = client.post("/predict", json=valid_input)
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 1.0


def test_batch_predict_response_time(client, valid_input):
    """Test: Tiempo de respuesta de /predict/batch es razonable"""
    import time
    
    batch_input = {
        "instances": [valid_input.copy() for _ in range(10)]
    }
    
    start = time.time()
    response = client.post("/predict/batch", json=batch_input)
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 2.0


def test_openapi_schema_available(client):
    """Test: OpenAPI schema está disponible"""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema


def test_docs_page_available(client):
    """Test: Página de documentación /docs está disponible"""
    response = client.get("/docs")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_page_available(client):
    """Test: Página de documentación /redoc está disponible"""
    response = client.get("/redoc")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]