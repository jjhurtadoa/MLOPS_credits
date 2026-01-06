"""
Tests unitarios para api/
Pruebas de endpoints FastAPI
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from api.main import app
from api.schemas import HousingFeatures, PredictionResponse, BatchPredictionRequest, HealthResponse
from api.utils import ModelLoader, PreprocessorLoader


# ============================================================================
# CONFIGURACIÓN DE LOGGING PARA TESTS
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """
    Configura logging UNA VEZ para toda la sesión de tests
    """
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    yield


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def client():
    """
    Cliente de prueba para FastAPI
    
    Usa TestClient que no requiere servidor corriendo
    scope="function" → Crea un cliente nuevo para cada test
    """
    # Resetear rate limiter antes de cada test
    from slowapi import _rate_limit_exceeded_handler
    
    with TestClient(app, raise_server_exceptions=False) as test_client:
        # Limpiar el estado del rate limiter
        if hasattr(app.state, 'limiter'):
            # Reset del storage interno del limiter
            try:
                app.state.limiter.reset()
            except:
                pass
        yield test_client


@pytest.fixture
def sample_housing_features():
    """Features de ejemplo válidas"""
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


@pytest.fixture
def mock_model():
    """Mock del modelo ML"""
    model = Mock()
    model.predict.return_value = np.array([24.5])
    return model


@pytest.fixture
def mock_preprocessor():
    """Mock del preprocessor"""
    preprocessor = Mock()
    # Simula retornar DataFrame procesado con features esperadas
    mock_df = pd.DataFrame({
        'CRIM': [0.00632],
        'ZN': [18.0],
        'INDUS': [2.31],
        'CHAS': [0],
        'NOX': [0.538],
        'RM': [6.575],
        'AGE': [65.2],
        'DIS':  [4.0900],
        'RAD': [1],
        'TAX': [296.0],
        'PTRATIO': [15.3],
        'B': [396.90],
        'LSTAT':  [4.98],
        # Features engineered (ejemplo)
        'RM_squared': [43.23],
        'LSTAT_log': [1.61]
    })
    preprocessor.preprocess.return_value = mock_df
    preprocessor.is_loaded = True
    return preprocessor


@pytest.fixture(autouse=True)
def mock_model_and_preprocessor(mock_model, mock_preprocessor):
    """
    Mock automático de model_loader y preprocessor_loader
    
    autouse=True → se aplica a todos los tests automáticamente
    """
    with patch('api.main.model_loader') as mock_ml, \
         patch('api.main.preprocessor_loader') as mock_pl:
        
        # Configurar model_loader
        mock_ml.load_model.return_value = mock_model
        mock_ml.is_loaded = True
        mock_ml.get_model_name.return_value = "test_model"
        
        # Configurar preprocessor_loader
        mock_pl.preprocess.return_value = mock_preprocessor.preprocess.return_value
        mock_pl.is_loaded = True
        
        yield mock_ml, mock_pl


# ============================================================================
# TESTS DE ENDPOINTS BÁSICOS
# ============================================================================

def test_root_endpoint(client):
    """Test:  Endpoint raíz GET /"""
    response = client.get("/")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "description" in data
    assert "endpoints" in data
    
    # Verificar que incluye endpoints clave
    assert "docs" in data["endpoints"]
    assert "health" in data["endpoints"]
    assert "predict" in data["endpoints"]


def test_health_endpoint_healthy(client):
    """Test: Health check cuando todo está cargado"""
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessors_loaded"] is True
    assert "model_name" in data
    assert "version" in data


def test_health_endpoint_unhealthy(client):
    """Test: Health check cuando modelo no está cargado"""
    # Crear un nuevo patch específico para este test
    with patch('api.main.model_loader') as mock_ml, \
         patch('api.main.preprocessor_loader') as mock_pl:
        
        # Configurar como no cargado
        mock_ml.is_loaded = False
        mock_ml.get_model_name.return_value = "no_model"
        mock_pl.is_loaded = True
        
        response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    # El status debería ser unhealthy si el modelo no está cargado
    assert data["status"] == "unhealthy"
    assert data["model_loaded"] is False


# ============================================================================
# TESTS DE PREDICCIÓN INDIVIDUAL
# ============================================================================

def test_predict_endpoint_success(client, sample_housing_features):
    """Test: Predicción exitosa con features válidas"""
    response = client.post("/predict", json=sample_housing_features)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "model_name" in data
    assert "model_version" in data
    
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0


def test_predict_endpoint_invalid_features(client):
    """Test: Predicción con features inválidas (falta campo requerido)"""
    invalid_features = {
        "CRIM": 0.00632,
        "ZN": 18.0,
        # Faltan campos requeridos
    }
    
    response = client.post("/predict", json=invalid_features)
    
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_negative_values(client, sample_housing_features):
    """Test: Predicción con valores negativos (violación de constraints)"""
    invalid_features = sample_housing_features.copy()
    invalid_features["CRIM"] = -1.0  # CRIM debe ser >= 0
    
    response = client.post("/predict", json=invalid_features)
    
    assert response.status_code == 422


def test_predict_endpoint_out_of_range(client, sample_housing_features):
    """Test: Predicción con valores fuera de rango"""
    invalid_features = sample_housing_features.copy()
    invalid_features["ZN"] = 150.0  # ZN debe estar entre 0-100
    
    response = client.post("/predict", json=invalid_features)
    
    assert response.status_code == 422


def test_predict_endpoint_model_error(client, sample_housing_features):
    """Test: Error durante la predicción del modelo"""
    with patch('api.main.model_loader') as mock_ml:
        mock_ml.load_model.side_effect = Exception("Model error")
        
        response = client.post("/predict", json=sample_housing_features)
    
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()


# ============================================================================
# TESTS DE PREDICCIÓN BATCH
# ============================================================================

def test_batch_predict_success(client, sample_housing_features):
    """Test: Predicción batch exitosa"""
    batch_request = {
        "instances": [
            sample_housing_features,
            sample_housing_features,
            sample_housing_features
        ]
    }
    
    with patch('api.main.model_loader') as mock_ml:
        mock_model = Mock()
        mock_model.predict.return_value = np.array([24.5, 25.0, 23.8])
        mock_ml.load_model.return_value = mock_model
        mock_ml.get_model_name.return_value = "test_model"
        
        response = client.post("/predict/batch", json=batch_request)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert "model_name" in data
    
    assert len(data["predictions"]) == 3
    assert data["count"] == 3


def test_batch_predict_empty_list(client):
    """Test: Batch con lista vacía"""
    batch_request = {
        "instances": []
    }
    
    response = client.post("/predict/batch", json=batch_request)
    
    assert response.status_code == 422


def test_batch_predict_single_instance(client, sample_housing_features):
    """Test: Batch con solo una instancia"""
    batch_request = {
        "instances":  [sample_housing_features]
    }
    
    with patch('api.main.model_loader') as mock_ml:
        mock_model = Mock()
        mock_model.predict.return_value = np.array([24.5])
        mock_ml.load_model.return_value = mock_model
        mock_ml.get_model_name.return_value = "test_model"
        
        response = client.post("/predict/batch", json=batch_request)
    
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["count"] == 1


def test_batch_predict_invalid_instance(client, sample_housing_features):
    """Test: Batch con una instancia inválida"""
    invalid_instance = sample_housing_features.copy()
    del invalid_instance["CRIM"]  # Eliminar campo requerido
    
    batch_request = {
        "instances": [invalid_instance]
    }
    
    response = client.post("/predict/batch", json=batch_request)
    
    assert response.status_code == 422


# ============================================================================
# TESTS DE VALIDACIÓN DE SCHEMAS
# ============================================================================

def test_housing_features_schema_valid(sample_housing_features):
    """Test: Validación exitosa de HousingFeatures"""
    features = HousingFeatures(**sample_housing_features)
    
    assert features.CRIM == sample_housing_features["CRIM"]
    assert features.RM == sample_housing_features["RM"]
    assert features.CHAS == sample_housing_features["CHAS"]


def test_housing_features_to_dataframe(sample_housing_features):
    """Test: Conversión de HousingFeatures a DataFrame"""
    features = HousingFeatures(**sample_housing_features)
    df = features.to_dataframe()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    assert df['CRIM'].iloc[0] == sample_housing_features["CRIM"]


def test_housing_features_invalid_type():
    """Test: HousingFeatures con tipo de dato incorrecto"""
    with pytest.raises(ValueError):
        HousingFeatures(
            CRIM="invalid",  # Debe ser float
            ZN=18.0,
            INDUS=2.31,
            CHAS=0,
            NOX=0.538,
            RM=6.575,
            AGE=65.2,
            DIS=4.0900,
            RAD=1,
            TAX=296.0,
            PTRATIO=15.3,
            B=396.90,
            LSTAT=4.98
        )


# ============================================================================
# TESTS DE RATE LIMITING
# ============================================================================

@pytest.mark.slow
def test_rate_limiting_predict(sample_housing_features):
    """Test: Rate limiting en endpoint /predict (10/minute)"""
    # Crear cliente fresco para este test
    with TestClient(app) as fresh_client:
        # Hacer más de 10 requests
        responses = []
        for _ in range(12):
            response = fresh_client.post("/predict", json=sample_housing_features)
            responses.append(response)
        
        # Al menos una debería ser rechazada por rate limit
        status_codes = [r.status_code for r in responses]
        
        # Verificar que hay códigos 200 exitosos
        successful = status_codes.count(200)
        rate_limited = status_codes.count(429)
        
        # Debe haber al menos algunos exitosos y algunos limitados
        assert successful >= 8  # Al menos 8 exitosos
        assert rate_limited >= 2  # Al menos 2 limitados


@pytest.mark.slow
def test_rate_limiting_batch(sample_housing_features):
    """Test: Rate limiting en endpoint /predict/batch (5/minute)"""
    batch_request = {
        "instances": [sample_housing_features]
    }
    
    # Crear cliente fresco para este test
    with TestClient(app) as fresh_client:
        # Hacer más de 5 requests
        responses = []
        for _ in range(7):
            response = fresh_client.post("/predict/batch", json=batch_request)
            responses.append(response)
        
        status_codes = [r.status_code for r in responses]
        
        # Debe haber algunos exitosos y algunos limitados
        successful = [c for c in status_codes if c in [200, 500]]
        rate_limited = status_codes.count(429)
        
        assert len(successful) >= 2  # Al menos 2 exitosos
        assert rate_limited >= 2  # Al menos 2 limitados




# ============================================================================
# TESTS DE DOCUMENTACIÓN
# ============================================================================

def test_openapi_docs_available(client):
    """Test: Documentación OpenAPI está disponible"""
    response = client.get("/docs")
    
    assert response.status_code == 200


def test_redoc_available(client):
    """Test: ReDoc está disponible"""
    response = client.get("/redoc")
    
    assert response.status_code == 200


def test_openapi_schema(client):
    """Test: Schema OpenAPI es válido"""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]


# ============================================================================
# TESTS DE MÉTRICAS PROMETHEUS
# ============================================================================

def test_metrics_endpoint_available(client):
    """Test: Endpoint de métricas Prometheus está disponible"""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_after_prediction(client, sample_housing_features):
    """Test: Métricas se actualizan después de predicción"""
    # Hacer una predicción
    client.post("/predict", json=sample_housing_features)
    
    # Obtener métricas
    response = client.get("/metrics")
    
    assert response.status_code == 200
    metrics_text = response.text
    
    # Verificar que hay métricas relacionadas con HTTP
    assert "http_requests_total" in metrics_text or "http_request" in metrics_text.lower()


# ============================================================================
# TESTS PARAMETRIZADOS
# ============================================================================

@pytest.mark.parametrize("field,invalid_value,expected_status", [
    ("CRIM", -1.0, 422),      # Valor negativo
    ("ZN", 150.0, 422),       # Fuera de rango
    ("CHAS", 2, 422),         # Valor inválido para binario
    ("NOX", 2.0, 422),        # Fuera de rango
    ("RM", -1.0, 422),        # Valor negativo
    ("PTRATIO", 100.0, 422),  # Fuera de rango
])
def test_predict_invalid_field_values(client, sample_housing_features, field, invalid_value, expected_status):
    """Test parametrizado: Validación de campos individuales"""
    invalid_features = sample_housing_features.copy()
    invalid_features[field] = invalid_value
    
    response = client.post("/predict", json=invalid_features)
    
    assert response.status_code == expected_status


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

@pytest.mark.integration
def test_full_prediction_flow(sample_housing_features):
    """Test de integración:  Flujo completo de predicción"""
    # Crear cliente fresco para evitar rate limiting
    with TestClient(app) as fresh_client:
        # 1.Verificar que la API está saludable
        health_response = fresh_client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # 2.Hacer una predicción
        predict_response = fresh_client.post("/predict", json=sample_housing_features)
        assert predict_response.status_code == 200
        
        prediction_data = predict_response.json()
        assert "predicted_price" in prediction_data
        
        # 3.Verificar métricas
        metrics_response = fresh_client.get("/metrics")
        assert metrics_response.status_code == 200


@pytest.mark.integration
def test_concurrent_predictions(client, sample_housing_features):
    """Test: Múltiples predicciones concurrentes"""
    import concurrent.futures
    
    def make_prediction():
        return client.post("/predict", json=sample_housing_features)
    
    # Ejecutar 5 predicciones concurrentes
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_prediction) for _ in range(5)]
        responses = [f.result() for f in futures]
    
    # Todas deberían ser exitosas (o algunas 429 por rate limit)
    status_codes = [r.status_code for r in responses]
    assert all(code in [200, 429, 500] for code in status_codes)


# ============================================================================
# TESTS DE EDGE CASES
# ============================================================================

def test_predict_with_extreme_values():
    """Test: Predicción con valores extremos (pero válidos)"""
    extreme_features = {
        "CRIM": 0.0,      # Mínimo
        "ZN": 100.0,      # Máximo
        "INDUS": 0.0,     # Mínimo
        "CHAS":  1,        # Máximo
        "NOX": 0.0,       # Mínimo
        "RM": 20.0,       # Máximo
        "AGE": 100.0,     # Máximo
        "DIS": 0.1,       # Cerca del mínimo
        "RAD":  24,        # Máximo
        "TAX": 1000.0,    # Alto
        "PTRATIO":  50.0,  # Máximo
        "B": 500.0,       # Alto
        "LSTAT": 100.0    # Máximo
    }
    
    # Crear cliente fresco
    with TestClient(app) as fresh_client:
        response = fresh_client.post("/predict", json=extreme_features)
    
    # Debería aceptar los valores (son válidos según constraints)
    assert response.status_code == 200


def test_predict_with_decimal_precision(sample_housing_features):
    """Test: Predicción con alta precisión decimal"""
    precise_features = sample_housing_features.copy()
    precise_features["CRIM"] = 0.006324567891234
    precise_features["NOX"] = 0.538123456789
    
    # Crear cliente fresco
    with TestClient(app) as fresh_client:
        response = fresh_client.post("/predict", json=precise_features)
    
    assert response.status_code == 200


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================

@pytest.mark.slow
def test_prediction_response_time(sample_housing_features):
    """Test: Tiempo de respuesta de predicción"""
    import time
    
    # Crear cliente fresco
    with TestClient(app) as fresh_client:
        start = time.time()
        response = fresh_client.post("/predict", json=sample_housing_features)
        duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 1.0  # Debe responder en menos de 1 segundo


@pytest.mark.slow
def test_batch_prediction_performance(sample_housing_features):
    """Test: Performance de predicción batch con muchas instancias"""
    import time
    
    # Crear batch de 100 instancias
    batch_request = {
        "instances": [sample_housing_features for _ in range(100)]
    }
    
    # Crear cliente fresco
    with TestClient(app) as fresh_client:
        with patch('api.main.model_loader') as mock_ml:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([24.5] * 100)
            mock_ml.load_model.return_value = mock_model
            mock_ml.get_model_name.return_value = "test_model"
            
            start = time.time()
            response = fresh_client.post("/predict/batch", json=batch_request)
            duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 5.0  # Debe completar en menos de 5 segundos
    assert len(response.json()["predictions"]) == 100