"""
Tests unitarios para src/data/
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import logging

from src.data.load_data import (
    load_raw_data,
    split_data,
    save_data,
    load_split_data
)


from src.utils.logger import setup_logging, reset_logging


# ============================================================================
# CONFIGURACIÓN DE LOGGING PARA TESTS
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """
    Configura logging UNA VEZ para toda la sesión de tests
    
    autouse=True → Se ejecuta automáticamente sin necesidad de especificar
    scope="session" → Una vez por toda la sesión de pytest
    """
    setup_logging(
        level=logging.WARNING,  # Solo warnings/errors en tests (menos ruido)
        log_file=None  # No guardar logs en tests
    )
    yield
    # Cleanup después de todos los tests
    reset_logging()


# ============================================================================
# FIXTURES (datos de prueba reutilizables)
# ============================================================================

@pytest.fixture
def sample_boston_data():
    """Crea un DataFrame de ejemplo con estructura de Boston Housing"""
    data = {
        'CRIM': [0.00632, 0.02731, 0.02729],
        'ZN': [18.0, 0.0, 0.0],
        'INDUS': [2.31, 7.07, 7.07],
        'CHAS':  [0, 0, 0],
        'NOX': [0.538, 0.469, 0.469],
        'RM': [6.575, 6.421, 7.185],
        'AGE': [65.2, 78.9, 61.1],
        'DIS': [4.0900, 4.9671, 4.9671],
        'RAD': [1, 2, 2],
        'TAX': [296, 242, 242],
        'PTRATIO': [15.3, 17.8, 17.8],
        'B':  [396.90, 396.90, 392.83],
        'LSTAT':  [4.98, 9.14, 4.03],
        'MEDV':  [24.0, 21.6, 34.7]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir():
    """Crea directorio temporal para tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Fixture para datos reales (opcional)
@pytest.fixture(scope="session")
def real_boston_data():
    """
    Carga datos reales de Boston Housing (si existen)
    Usa scope='session' para cargar solo una vez
    """
    real_path = Path('data/raw/HousingData.csv')
    if real_path.exists():
        return pd.read_csv(real_path)
    return None



# ============================================================================
# TESTS EDGE CASES
# ============================================================================


def test_split_data_single_row():
    """Test:  Split con solo una fila (edge case)"""
    df = pd.DataFrame({
        'A': [1], 'B': [2], 'MEDV': [3]
    })
    
    with pytest.raises(ValueError):
        # No se puede dividir 1 fila
        split_data(df, test_size=0.2)


def test_save_data_with_special_characters(sample_boston_data, temp_data_dir):
    """Test: Guardar con caracteres especiales en path"""
    X_train, X_test, y_train, y_test = split_data(sample_boston_data)
    
    # Usar Path más simple
    special_dir = Path(temp_data_dir) / 'test dir'
    
    # No debería lanzar excepción
    save_data(X_train, X_test, y_train, y_test, output_dir=str(special_dir))
    
    # Verificar que se creó algo
    assert special_dir.exists()
    assert len(list(special_dir.glob('*.csv'))) == 2  # train.csv y test.csv


def test_load_data_with_missing_values(temp_data_dir):
    """Test: Cargar datos con valores faltantes"""
    df_with_na = pd.DataFrame({
        'A': [1, None, 3],
        'B': [4, 5, None],
        'MEDV':  [7, 8, 9]
    })
    
    test_file = Path(temp_data_dir) / 'with_na.csv'
    df_with_na.to_csv(test_file, index=False)
    
    df_loaded = load_raw_data(str(test_file))
    assert df_loaded.isnull().sum().sum() > 0  # Tiene NAs


# ============================================================================
# TESTS PARAMETRIZADOS
# ============================================================================

@pytest.mark.parametrize("test_size,expected_train_pct", [
    (0.2, 0.8),
    (0.3, 0.7),
    (0.5, 0.5),
])
def test_split_sizes(sample_boston_data, test_size, expected_train_pct):
    """Test parametrizado:  Diferentes tamaños de split"""
    df_large = pd.concat([sample_boston_data] * 100, ignore_index=True)
    
    X_train, X_test, y_train, y_test = split_data(
        df_large, test_size=test_size, random_state=42
    )
    
    total = len(X_train) + len(X_test)
    train_pct = len(X_train) / total
    
    assert abs(train_pct - expected_train_pct) < 0.05  # Tolerancia 5%


# ============================================================================
# TESTS DE PERFORMANCE 
# ============================================================================

@pytest.mark.slow
def test_split_large_dataset():
    """Test con dataset grande (marcar como slow)"""
    # Crear dataset grande
    df_large = pd.DataFrame({
        f'feature_{i}': range(100000) for i in range(13)
    })
    df_large['MEDV'] = range(100000)
    
    import time
    start = time.time()
    
    X_train, X_test, y_train, y_test = split_data(df_large, test_size=0.2)
    
    duration = time.time() - start
    
    assert len(X_train) == 80000
    assert duration < 5.0  # Debe completar en menos de 5 segundos


# ============================================================================
# TESTS CON DATOS REALES
# ============================================================================

@pytest.mark.integration
def test_with_real_data(real_boston_data):
    """Test de integración con datos reales"""
    if real_boston_data is None: 
        pytest.skip("Datos reales no disponibles")
    
    # Split
    X_train, X_test, y_train, y_test = split_data(
        real_boston_data, test_size=0.2, random_state=42
    )
    
    # Validaciones
    assert len(X_train) == 404
    assert len(X_test) == 102
    assert X_train.shape[1] == 13
    
    # Verificar que no hay data leakage
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0


def test_load_split_data_roundtrip(sample_boston_data, temp_data_dir):
    """Guarda splits y los carga con `load_split_data`, verificando igualdad."""
    # Generar split
    X_train, X_test, y_train, y_test = split_data(sample_boston_data, random_state=42)

    # Guardar en directorio temporal
    save_data(X_train, X_test, y_train, y_test, output_dir=temp_data_dir)

    # Cargar usando la función a testear
    X_t, X_te, y_t, y_te = load_split_data(data_dir=temp_data_dir)

    # Comparar shapes y columnas
    assert X_t.shape == X_train.shape
    assert X_te.shape == X_test.shape
    assert list(X_t.columns) == list(X_train.columns)
    assert list(X_te.columns) == list(X_test.columns)

    # Comparar contenido (reseteando índices)
    pd.testing.assert_frame_equal(X_t.reset_index(drop=True), X_train.reset_index(drop=True))
    pd.testing.assert_frame_equal(X_te.reset_index(drop=True), X_test.reset_index(drop=True))
    pd.testing.assert_series_equal(y_t.reset_index(drop=True), y_train.reset_index(drop=True))
    pd.testing.assert_series_equal(y_te.reset_index(drop=True), y_test.reset_index(drop=True))


