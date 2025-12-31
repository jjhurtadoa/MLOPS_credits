"""
Tests para src/train/

Ejecutar: 
    pytest tests/test_training.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
import joblib

from src.train.train_model import ModelTrainer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_train_data():
    """Datos sintéticos de entrenamiento"""
    np.random.seed(42)
    n = 100
    
    X_train = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n)
    })
    
    # Target con relación lineal + ruido
    y_train = (2 * X_train['feature1'] + 
               3 * X_train['feature2'] - 
               1 * X_train['feature3'] + 
               np.random.randn(n) * 0.5)
    
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Datos sintéticos de test"""
    np.random.seed(43)
    n = 30
    
    X_test = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3':  np.random.randn(n)
    })
    
    y_test = (2 * X_test['feature1'] + 
              3 * X_test['feature2'] - 
              1 * X_test['feature3'] + 
              np.random.randn(n) * 0.5)
    
    return X_test, y_test


@pytest.fixture
def minimal_config(tmp_path):
    """Config mínima para tests CON CAMPO MODULE"""
    config = {
        'models': {
            'random_forest': {
                'enabled': True,
                'module': 'sklearn.ensemble',        # ← NUEVO:  Requerido para import dinámico
                'class':   'RandomForestRegressor',
                'params': {
                    'n_estimators': 10,  # Pocos para tests rápidos
                    'max_depth': 3,
                    'random_state': 42,
                    'n_jobs': 1  # No paralelo en tests
                }
            }
        },
        'training': {
            'data_dir': str(tmp_path / 'data'),
            'target_column': 'target',
            'model_dir': str(tmp_path / 'models'),
            'mlflow':   {
                'experiment_name':   'test_experiment',
                'tracking_uri':  str(tmp_path / 'mlruns'),
                'run_name_prefix': 'test'
            },
            'metrics': ['rmse', 'mae', 'r2']
        },
        'validation': {
            'cv_folds': 3,  # Menos folds para tests rápidos
            'cv_scoring': 'neg_root_mean_squared_error'
        }
    }
    
    return config


# ============================================================================
# TESTS
# ============================================================================

def test_model_trainer_init(minimal_config):
    """Test: Inicialización de ModelTrainer"""
    trainer = ModelTrainer(minimal_config)
    
    assert trainer.config == minimal_config
    assert trainer.models_ == {}
    assert trainer.metrics_ == {}


def test_calculate_metrics(minimal_config, sample_train_data):
    """Test: Cálculo de métricas"""
    _, y_train = sample_train_data
    
    trainer = ModelTrainer(minimal_config)
    
    # Predicciones simuladas (perfectas)
    y_pred = y_train.values
    
    metrics = trainer.calculate_metrics(y_train, y_pred)
    
    # Predicciones perfectas → RMSE ≈ 0, R² ≈ 1
    assert metrics['rmse'] == pytest.approx(0, abs=1e-10)
    assert metrics['mae'] == pytest.approx(0, abs=1e-10)
    assert metrics['r2'] == pytest.approx(1.0, abs=1e-10)
    assert 'mape' in metrics


def test_train_single_model(minimal_config, sample_train_data, sample_test_data):
    """Test: Entrenar un modelo"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    trainer = ModelTrainer(minimal_config)
    
    model_config = minimal_config['models']['random_forest']
    
    result = trainer.train_model(
        'random_forest',
        model_config,
        X_train, y_train,
        X_test, y_test
    )
    
    # Validaciones
    assert result is not None
    assert 'model' in result
    assert 'metadata' in result
    
    # Metadata debe tener métricas
    metadata = result['metadata']
    assert 'train_metrics' in metadata
    assert 'test_metrics' in metadata
    assert 'cv_results' in metadata
    
    # Métricas deben ser razonables
    assert metadata['test_metrics']['r2'] > 0  # Al menos algo de fit
    assert metadata['test_metrics']['rmse'] > 0


def test_cross_validation(minimal_config, sample_train_data):
    """Test: Cross-validation"""
    from sklearn.ensemble import RandomForestRegressor
    
    X_train, y_train = sample_train_data
    
    trainer = ModelTrainer(minimal_config)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    cv_results = trainer.cross_validate_model(model, X_train, y_train)
    
    # Validaciones
    assert 'cv_mean' in cv_results
    assert 'cv_std' in cv_results
    assert 'cv_min' in cv_results
    assert 'cv_max' in cv_results
    
    # Valores razonables
    assert cv_results['cv_mean'] > 0
    assert cv_results['cv_std'] >= 0


def test_train_all_models(minimal_config, sample_train_data, sample_test_data):
    """Test: Entrenar todos los modelos"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    trainer = ModelTrainer(minimal_config)
    
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Debe haber entrenado al menos un modelo
    assert len(trainer.models_) > 0
    assert len(trainer.metrics_) > 0
    
    # Verificar que random_forest está entrenado
    assert 'random_forest' in trainer.models_
    assert 'random_forest' in trainer.metrics_


def test_get_best_model(minimal_config, sample_train_data, sample_test_data):
    """Test: Obtener mejor modelo"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    trainer = ModelTrainer(minimal_config)
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    best_name, best_model, best_metadata = trainer.get_best_model(metric='test_rmse')
    
    # Validaciones
    assert best_name in trainer.models_
    assert best_model is not None
    assert best_metadata is not None
    assert 'test_metrics' in best_metadata


def test_compare_models(minimal_config, sample_train_data, sample_test_data):
    """Test: Comparar modelos"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    trainer = ModelTrainer(minimal_config)
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    comparison_df = trainer.compare_models()
    
    # Validaciones
    assert isinstance(comparison_df, pd.DataFrame)
    assert len(comparison_df) > 0
    assert 'model' in comparison_df.columns
    assert 'test_rmse' in comparison_df.columns


def test_model_persistence(minimal_config, sample_train_data, sample_test_data, tmp_path):
    """Test: Guardar y cargar modelo"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    trainer = ModelTrainer(minimal_config)
    
    result = trainer.train_model(
        'random_forest',
        minimal_config['models']['random_forest'],
        X_train, y_train,
        X_test, y_test
    )
    
    model = result['model']
    model_path = tmp_path / 'test_model.pkl'
    
    # Guardar
    joblib.dump(model, model_path)
    
    # Cargar
    loaded_model = joblib.load(model_path)
    
    # Verificar que funciona igual
    pred_original = model.predict(X_test)
    pred_loaded = loaded_model.predict(X_test)
    
    np.testing.assert_array_equal(pred_original, pred_loaded)


def test_disabled_model_is_skipped(minimal_config, sample_train_data, sample_test_data):
    """Test: Modelos deshabilitados no se entrenan"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    # Deshabilitar modelo
    minimal_config['models']['random_forest']['enabled'] = False
    
    trainer = ModelTrainer(minimal_config)
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # No debe haber entrenado nada
    assert len(trainer.models_) == 0
    assert len(trainer.metrics_) == 0


@pytest.mark.integration
def test_full_training_pipeline(tmp_path):
    """Test de integración: Pipeline completo"""
    # Crear datos de prueba
    np.random.seed(42)
    n_train, n_test = 100, 30
    
    X_train = pd.DataFrame(np.random.randn(n_train, 5), columns=[f'f{i}' for i in range(5)])
    y_train = X_train.sum(axis=1) + np.random.randn(n_train) * 0.5
    
    X_test = pd.DataFrame(np.random.randn(n_test, 5), columns=[f'f{i}' for i in range(5)])
    y_test = X_test.sum(axis=1) + np.random.randn(n_test) * 0.5
    
    # Guardar datos
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv(data_dir / 'train_processed.csv', index=False)
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(data_dir / 'test_processed.csv', index=False)
    
    # Config
    config = {
        'models': {
            'random_forest': {
                'enabled': True,
                'module': 'sklearn.ensemble', 
                'class': 'RandomForestRegressor',
                'params': {'n_estimators': 10, 'random_state': 42}
            }
        },
        'training': {
            'data_dir': str(data_dir),
            'target_column': 'target',
            'model_dir': str(tmp_path / 'models'),
            'mlflow': {
                'experiment_name': 'test',
                'tracking_uri': str(tmp_path / 'mlruns'),
                'run_name_prefix': 'test'
            }
        },
        'validation':  {
            'cv_folds': 3,
            'cv_scoring':  'neg_root_mean_squared_error'
        }
    }
    
    # Ejecutar pipeline
    trainer = ModelTrainer(config)
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = trainer.load_data()
    trainer.train_all_models(X_train_loaded, y_train_loaded, X_test_loaded, y_test_loaded)
    
    # Validar outputs
    model_dir = tmp_path / 'models'
    assert (model_dir / 'random_forest_model.pkl').exists()
    assert (model_dir / 'random_forest_metadata.yaml').exists()
    
    # Validar que modelo funciona
    model = joblib.load(model_dir / 'random_forest_model.pkl')
    predictions = model.predict(X_test_loaded)
    assert len(predictions) == len(y_test_loaded)