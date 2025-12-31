"""
Tests unitarios para src/preprocess/

Ejecutar:  
    pytest tests/test_preprocessing.py -v
    pytest tests/test_preprocessing.py -v --cov=src.preprocess
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.preprocess.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    CorrelationReducer,
    FeatureScaler,
    DataPreprocessor
)

from src.preprocess.build_features import (
    InteractionFeatureCreator,
    RealEstateDomainFeatures,
    PolynomialFeatureCreator,
    FeatureEngineer
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Crea DataFrame de ejemplo con valores faltantes y outliers"""
    np.random.seed(42)
    data = {
        'A': [1.0, 2.0, np.nan, 4.0, 5.0, 100.0],  # Con NaN y outlier
        'B': [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
        'C': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Sin varianza
        'D': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # Normal
    }
    return pd.DataFrame(data)


@pytest.fixture
def boston_like_data():
    """Crea datos similares a Boston Housing (después de preprocessing)"""
    np.random.seed(42)
    n = 100
    
    # Simula tus features DESPUÉS de preprocessing
    # (sin RAD ni B que eliminaste)
    data = {
        'CRIM': np.random.exponential(3, n),
        'ZN': np.random.uniform(0, 100, n),
        'INDUS': np.random.uniform(0, 28, n),
        'CHAS': np.random.choice([0, 1], n),
        'NOX': np.random.uniform(0.3, 0.9, n),
        'RM': np.random.normal(6.5, 1, n),
        'AGE': np.random.uniform(0, 100, n),
        'DIS': np.random.uniform(1, 12, n),
        'TAX': np.random.randint(200, 700, n),
        'PTRATIO': np.random.uniform(12, 22, n),
        'LSTAT': np.random.uniform(1, 38, n),
    }
    
    df = pd.DataFrame(data)
    
    # Agregar algunos NaNs
    df.loc[df.sample(5, random_state=42).index, 'CRIM'] = np.nan
    df.loc[df.sample(5, random_state=43).index, 'AGE'] = np.nan
    
    return df


# ============================================================================
# TESTS:  MissingValueHandler
# ============================================================================

def test_missing_value_handler_mean(sample_data):
    """Test:  Imputación con media"""
    config = {'mean': ['A', 'B']}
    handler = MissingValueHandler(config)
    
    # Fit
    handler.fit(sample_data)
    
    # Transform
    result = handler.transform(sample_data)
    
    # Validaciones
    assert result['A'].isnull().sum() == 0, "No deben quedar NAs en A"
    assert result['B'].isnull().sum() == 0, "No deben quedar NAs en B"
    
    # Verificar que usó la media
    expected_mean_A = sample_data['A'].mean()
    assert result.loc[2, 'A'] == pytest.approx(expected_mean_A, rel=0.01)


def test_missing_value_handler_median():
    """Test: Imputación con mediana"""
    df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 100.0]})  # Outlier 100
    config = {'median': ['A']}
    
    handler = MissingValueHandler(config)
    result = handler.fit_transform(df)
    
    # Debe usar mediana, no media (resistente a outliers)
    expected_median = df['A'].median()  # = 2.0
    assert result.loc[2, 'A'] == pytest.approx(expected_median, rel=0.01)


def test_missing_value_handler_preserves_non_na():
    """Test: No modifica valores no faltantes"""
    df = pd.DataFrame({'A': [1.0, np.nan, 3.0]})
    config = {'mean':  ['A']}
    
    handler = MissingValueHandler(config)
    result = handler.fit_transform(df)
    
    assert result.loc[0, 'A'] == 1.0
    assert result.loc[2, 'A'] == 3.0


# ============================================================================
# TESTS:  OutlierHandler
# ============================================================================

def test_outlier_handler_log_transform(sample_data):
    """Test: Log transform reduce outliers"""
    config = {'log_transform': ['A']}
    handler = OutlierHandler(config)
    
    result = handler.fit_transform(sample_data)
    
    # Validar que aplicó log (reduce magnitud de outliers)
    assert result['A'].max() < sample_data['A'].max()


def test_outlier_handler_iqr_capping(sample_data):
    """Test: IQR capping limita outliers"""
    config = {'capping_iqr': ['A']}
    handler = OutlierHandler(config, iqr_factor=1.5)
    
    handler.fit(sample_data)
    result = handler.transform(sample_data)
    
    # El outlier (100) debe ser capped
    assert result['A'].max() < 100


def test_outlier_handler_fit_transform_consistency():
    """Test: fit_transform da mismo resultado que fit + transform"""
    df = pd.DataFrame({'A': [1, 2, 3, 100, 5]})
    config = {'capping_iqr': ['A']}
    
    handler1 = OutlierHandler(config)
    result1 = handler1.fit_transform(df)
    
    handler2 = OutlierHandler(config)
    handler2.fit(df)
    result2 = handler2.transform(df)
    
    pd.testing.assert_frame_equal(result1, result2)


# ============================================================================
# TESTS: CorrelationReducer
# ============================================================================

def test_correlation_reducer_removes_columns():
    """Test: Elimina columnas configuradas"""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],  # Correlacionada con A
        'C': [5, 4, 3, 2, 1]
    })
    
    config = {'remove':  ['B'], 'no_action': ['A', 'C']}
    reducer = CorrelationReducer(config)
    
    result = reducer.fit_transform(df)
    
    assert 'A' in result.columns
    assert 'B' not in result.columns, "B debería ser eliminada"
    assert 'C' in result.columns


def test_correlation_reducer_handles_missing_columns():
    """Test: No falla si columna a eliminar no existe"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    config = {'remove':  ['C', 'D'], 'no_action': []}  # C y D no existen
    
    reducer = CorrelationReducer(config)
    result = reducer.fit_transform(df)
    
    # No debe fallar, solo retornar df sin cambios
    assert result.shape == df.shape


def test_correlation_reducer_get_feature_names_out():
    """Test: get_feature_names_out retorna nombres correctos"""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    config = {'remove': ['B'], 'no_action': []}
    
    reducer = CorrelationReducer(config)
    reducer.fit(df)
    
    feature_names = reducer.get_feature_names_out()
    
    assert 'A' in feature_names
    assert 'B' not in feature_names
    assert 'C' in feature_names


# ============================================================================
# TESTS: FeatureScaler
# ============================================================================

def test_feature_scaler_robust():
    """Test: RobustScaler funciona correctamente"""
    df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [10, 20, 30, 40]})
    
    scaler = FeatureScaler(scaler_type='RobustScaler')
    result = scaler.fit_transform(df)
    
    # Debe estar escalado
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    
    # RobustScaler centra en la mediana
    assert result['A'].median() == pytest.approx(0, abs=0.5)


def test_feature_scaler_standard():
    """Test: StandardScaler funciona correctamente"""
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    
    scaler = FeatureScaler(scaler_type='StandardScaler')
    result = scaler.fit_transform(df)
    
    # StandardScaler → media 0, std 1
    assert result['A'].mean() == pytest.approx(0, abs=0.01)
    assert result['A'].std(ddof=0) == pytest.approx(1, abs=0.01)


# ============================================================================
# TESTS:  DataPreprocessor (Pipeline Completo)
# ============================================================================

def test_data_preprocessor_full_pipeline(boston_like_data):
    """Test: Pipeline completo de preprocessing"""
    config = {
        'handling_na': {
            'mean': [],
            'median': ['CRIM', 'AGE'],
            'most_frequent': [],
            'no_action': []
        },
        'handling_outliers': {
            'log_transform': ['CRIM'],
            'capping_iqr': [],
            'capping_winsor': [],
            'remove': [],
            'no_action': []
        },
        'handling_correlated_features': {
            'remove':  [],
            'no_action': []
        },
        'scaler': 'RobustScaler'
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Fit y transform
    X_transformed = preprocessor.fit_transform(boston_like_data)
    
    # Validaciones
    assert X_transformed.isnull().sum().sum() == 0, "No deben quedar NAs"
    assert X_transformed.shape[0] == boston_like_data.shape[0], "Mismas filas"
    assert isinstance(X_transformed, pd.DataFrame)


def test_data_preprocessor_transform_consistency(boston_like_data):
    """Test: Transform produce resultados consistentes"""
    config = {
        'handling_na':  {'mean': [], 'median': ['CRIM'], 'most_frequent': [], 'no_action': []},
        'handling_outliers':  {'log_transform': [], 'capping_iqr': [], 'capping_winsor': [], 'remove': [], 'no_action': []},
        'handling_correlated_features': {'remove': [], 'no_action': []},
        'scaler': 'StandardScaler'
    }
    
    preprocessor = DataPreprocessor(config)
    preprocessor.fit(boston_like_data)
    
    # Transform múltiples veces debe dar mismo resultado
    result1 = preprocessor.transform(boston_like_data)
    result2 = preprocessor.transform(boston_like_data)
    
    pd.testing.assert_frame_equal(result1, result2)


def test_data_preprocessor_save_load(boston_like_data, tmp_path):
    """Test: Guardar y cargar preprocessor"""
    config = {
        'handling_na': {'mean': [], 'median': ['CRIM'], 'most_frequent': [], 'no_action':  []},
        'handling_outliers': {'log_transform': [], 'capping_iqr': [], 'capping_winsor': [], 'remove': [], 'no_action': []},
        'handling_correlated_features': {'remove': [], 'no_action': []},
        'scaler': 'RobustScaler'
    }
    
    preprocessor = DataPreprocessor(config)
    preprocessor.fit(boston_like_data)
    
    # Guardar en directorio temporal
    save_path = tmp_path / 'preprocessor.pkl'
    preprocessor.save(save_path)
    
    assert save_path.exists(), "Archivo debe existir"
    
    # Cargar
    loaded = DataPreprocessor.load(save_path)
    
    # Verificar que funciona igual
    result_original = preprocessor.transform(boston_like_data)
    result_loaded = loaded.transform(boston_like_data)
    
    pd.testing.assert_frame_equal(result_original, result_loaded)


# ============================================================================
# TESTS: InteractionFeatureCreator
# ============================================================================

def test_interaction_feature_product():
    """Test: Feature de producto"""
    df = pd.DataFrame({'A': [2, 4, 6], 'B':  [3, 5, 7]})
    
    interactions = [
        {'type': 'product', 'features': ['A', 'B'], 'name': 'A_x_B'}
    ]
    
    creator = InteractionFeatureCreator(interactions)
    result = creator.fit_transform(df)
    
    assert 'A_x_B' in result.columns
    assert result['A_x_B'].tolist() == [6, 20, 42]


def test_interaction_feature_ratio():
    """Test: Feature de ratio"""
    df = pd.DataFrame({'A': [10, 20, 30], 'B': [2, 4, 5]})
    
    interactions = [
        {'type': 'ratio', 'features': ['A', 'B'], 'name': 'A_per_B'}
    ]
    
    creator = InteractionFeatureCreator(interactions)
    result = creator.fit_transform(df)
    
    assert 'A_per_B' in result.columns
    assert result['A_per_B'].tolist() == [5.0, 5.0, 6.0]


def test_interaction_feature_missing_columns():
    """Test: Omite interacciones con columnas faltantes"""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    interactions = [
        {'type': 'product', 'features': ['A', 'C'], 'name': 'A_x_C'}  # C no existe
    ]
    
    creator = InteractionFeatureCreator(interactions)
    result = creator.fit_transform(df)
    
    # No debe crear A_x_C (falta columna C)
    assert 'A_x_C' not in result.columns
    assert result.shape[1] == df.shape[1]  # Sin features nuevas


# ============================================================================
# TESTS:  RealEstateDomainFeatures
# ============================================================================

def test_real_estate_domain_features_creates_features(boston_like_data):
    """Test:  Creación de domain features"""
    creator = RealEstateDomainFeatures()
    
    result = creator.fit_transform(boston_like_data)
    
    # Verificar que se crearon features
    assert result.shape[1] > boston_like_data.shape[1], "Debe agregar features"
    
    # Features que deben existir (tienes todas las columnas necesarias)
    assert 'rooms_per_age' in result.columns
    assert 'tax_per_room' in result.columns
    assert 'socioeconomic_index' in result.columns
    assert 'property_quality' in result.columns


def test_real_estate_domain_features_no_rad():
    """Test: No crea accessibility_score si falta RAD"""
    # Datos SIN RAD (como tu caso real)
    df = pd.DataFrame({
        'RM': [6.5, 5.2, 7.1],
        'AGE': [65.2, 78.9, 45.8],
        'DIS': [3.2, 5.1, 2.8],
        'LSTAT': [4.98, 9.14, 4.03]
    })
    
    creator = RealEstateDomainFeatures()
    result = creator.fit_transform(df)
    
    # NO debe crear accessibility_score (requiere RAD)
    assert 'accessibility_score' not in result.columns
    
    # SÍ debe crear rooms_per_age (tiene RM y AGE)
    assert 'rooms_per_age' in result.columns


def test_real_estate_domain_features_calculations():
    """Test: Validar cálculos de domain features"""
    df = pd.DataFrame({
        'RM': [6.0],
        'AGE': [50.0],
        'TAX': [300.0],
        'LSTAT': [10.0],
        'PTRATIO': [15.0],
        'CRIM': [0.5],
        'INDUS': [5.0],
        'NOX': [0.5]
    })
    
    creator = RealEstateDomainFeatures()
    result = creator.fit_transform(df)
    
    # Validar cálculos
    assert result['rooms_per_age'].iloc[0] == pytest.approx(6.0 / 51.0, rel=0.01)  # RM / (AGE + 1)
    assert result['tax_per_room'].iloc[0] == pytest.approx(300.0 / 6.0, rel=0.01)  # TAX / RM
    assert result['property_quality'].iloc[0] == pytest.approx(6.0 * 0.9, rel=0.01)  # RM * (1 - 10/100)


# ============================================================================
# TESTS: FeatureEngineer (Pipeline Completo)
# ============================================================================

def test_feature_engineer_domain_features_only(boston_like_data):
    """Test: FeatureEngineer solo con domain features"""
    config = {
        'domain_features': True,
        'interactions': []
    }
    
    engineer = FeatureEngineer(config)
    result = engineer.fit_transform(boston_like_data)
    
    # Verificar que agregó features
    assert result.shape[1] > boston_like_data.shape[1]
    assert 'rooms_per_age' in result.columns


def test_feature_engineer_with_interactions(boston_like_data):
    """Test: FeatureEngineer con interacciones"""
    config = {
        'domain_features': True,
        'interactions': [
            {'type': 'product', 'features': ['RM', 'LSTAT'], 'name': 'RM_x_LSTAT'}
        ]
    }
    
    engineer = FeatureEngineer(config)
    result = engineer.fit_transform(boston_like_data)
    
    # Debe tener domain features + interacciones
    assert 'RM_x_LSTAT' in result.columns
    assert 'rooms_per_age' in result.columns


def test_feature_engineer_empty_config(boston_like_data):
    """Test: FeatureEngineer con config vacío"""
    config = {
        'domain_features': False,
        'interactions': []
    }
    
    engineer = FeatureEngineer(config)
    result = engineer.fit_transform(boston_like_data)
    
    # No debe agregar features
    assert result.shape[1] == boston_like_data.shape[1]


def test_feature_engineer_save_load(boston_like_data, tmp_path):
    """Test: Guardar y cargar FeatureEngineer"""
    config = {
        'domain_features': True,
        'interactions': []
    }
    
    engineer = FeatureEngineer(config)
    engineer.fit(boston_like_data)
    
    # Guardar
    save_path = tmp_path / 'feature_engineer.pkl'
    engineer.save(save_path)
    
    assert save_path.exists()
    
    # Cargar
    loaded = FeatureEngineer.load(save_path)
    
    # Verificar que funciona igual
    result_original = engineer.transform(boston_like_data)
    result_loaded = loaded.transform(boston_like_data)
    
    pd.testing.assert_frame_equal(result_original, result_loaded)


# ============================================================================
# TESTS:  Integración con Config Real
# ============================================================================

@pytest.mark.integration
def test_with_real_preprocessing_config(boston_like_data):
    """Test: Pipeline completo con config real"""
    # Tu config real
    config = {
        'feature_processing': {
            'handling_na': {
                'mean': [],
                'median': ['CRIM', 'ZN', 'INDUS', 'AGE', 'LSTAT'],
                'most_frequent': ['CHAS'],
                'no_action': []
            },
            'handling_outliers': {
                'log_transform': ['CRIM', 'ZN', 'LSTAT'],
                'capping_iqr': ['RM'],
                'capping_winsor': ['PTRATIO'],
                'remove':  [],
                'no_action': []
            },
            'handling_correlated_features': {
                'remove': [],
                'no_action': []
            },
            'scaler': 'RobustScaler'
        },
        'feature_engineering': {
            'domain_features': True,
            'interactions': []
        }
    }
    
    # Preprocessing
    preprocessor = DataPreprocessor(config['feature_processing'])
    X_prep = preprocessor.fit_transform(boston_like_data)
    
    # Feature engineering
    engineer = FeatureEngineer(config['feature_engineering'])
    X_final = engineer.fit_transform(X_prep)
    
    # Validaciones
    assert X_final.isnull().sum().sum() == 0, "No deben quedar NAs"
    assert X_final.shape[0] == boston_like_data.shape[0], "Mismas filas"
    assert X_final.shape[1] > boston_like_data.shape[1], "Debe tener más columnas"