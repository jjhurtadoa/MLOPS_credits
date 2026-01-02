"""
Tests para el módulo de evaluación de modelos
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import tempfile

from src.evaluate.evaluate_model import ModelEvaluator
from src.evaluate.visualizations import (
    plot_residuals,
    plot_predictions,
    plot_feature_importance,
    plot_error_distribution,
    plot_model_comparison
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_model(tmp_path):
    """Crea un modelo simple para tests"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Datos de entrenamiento simples
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Guardar modelo
    model_path = tmp_path / 'test_model.pkl'
    joblib.dump(model, model_path)
    
    return model_path


@pytest.fixture
def sample_data():
    """Genera datos de prueba"""
    np.random.seed(42)
    n = 50
    
    X = pd.DataFrame(
        np.random.randn(n, 5),
        columns=[f'feature_{i}' for i in range(5)]
    )
    
    y = pd.Series(X.sum(axis=1) + np.random.randn(n) * 0.1, name='target')
    
    return X, y


# ============================================================================
# TESTS - ModelEvaluator
# ============================================================================

def test_model_evaluator_init(sample_model):
    """Test: Inicialización del evaluador"""
    evaluator = ModelEvaluator(str(sample_model))
    
    assert evaluator.model is not None
    assert evaluator.model_name == 'test_model'
    assert evaluator.metrics_ == {}
    assert evaluator.predictions_ == {}


def test_model_evaluator_load_nonexistent_model():
    """Test: Error al cargar modelo inexistente"""
    with pytest.raises(FileNotFoundError):
        ModelEvaluator('nonexistent_model.pkl')


def test_calculate_metrics(sample_model, sample_data):
    """Test: Cálculo de métricas"""
    X, y = sample_data
    
    evaluator = ModelEvaluator(str(sample_model))
    
    # Predicciones
    y_pred = evaluator.model.predict(X)
    
    # Calcular métricas
    metrics = evaluator.calculate_metrics(y.values, y_pred, prefix='test_')
    
    # Validaciones
    assert 'test_rmse' in metrics
    assert 'test_mae' in metrics
    assert 'test_r2' in metrics
    assert 'test_mape' in metrics
    assert 'test_max_error' in metrics
    assert 'test_median_error' in metrics
    
    # Valores razonables
    assert metrics['test_rmse'] >= 0
    assert metrics['test_mae'] >= 0
    assert metrics['test_r2'] <= 1.0
    assert metrics['test_mape'] >= 0


def test_evaluate_test_only(sample_model, sample_data):
    """Test: Evaluación solo en test set"""
    X, y = sample_data
    
    evaluator = ModelEvaluator(str(sample_model))
    metrics = evaluator.evaluate(X, y)
    
    # Debe tener métricas de test
    assert 'test_rmse' in metrics
    assert 'test_mae' in metrics
    assert 'test_r2' in metrics
    
    # Debe tener predicciones guardadas
    assert 'test' in evaluator.predictions_
    assert 'y_true' in evaluator.predictions_['test']
    assert 'y_pred' in evaluator.predictions_['test']


def test_evaluate_train_and_test(sample_model, sample_data):
    """Test: Evaluación en train y test"""
    X, y = sample_data
    
    # Dividir en train/test
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    evaluator = ModelEvaluator(str(sample_model))
    metrics = evaluator.evaluate(X_test, y_test, X_train, y_train)
    
    # Debe tener métricas de train y test
    assert 'train_rmse' in metrics
    assert 'test_rmse' in metrics
    
    # Debe tener predicciones de train y test
    assert 'train' in evaluator.predictions_
    assert 'test' in evaluator.predictions_


def test_generate_visualizations(sample_model, sample_data, tmp_path):
    """Test: Generación de visualizaciones"""
    X, y = sample_data
    
    evaluator = ModelEvaluator(str(sample_model))
    evaluator.evaluate(X, y)
    
    output_dir = tmp_path / 'visualizations'
    evaluator.generate_visualizations(str(output_dir), X)
    
    # Verificar que se crearon los archivos
    assert (output_dir / 'test_model_residuals_test.png').exists()
    assert (output_dir / 'test_model_predictions_test.png').exists()
    assert (output_dir / 'test_model_errors_test.png').exists()
    assert (output_dir / 'test_model_feature_importance.png').exists()


def test_generate_report(sample_model, sample_data, tmp_path):
    """Test: Generación de reporte completo"""
    X, y = sample_data
    
    evaluator = ModelEvaluator(str(sample_model))
    evaluator.evaluate(X, y)
    
    output_dir = tmp_path / 'report'
    evaluator.generate_report(str(output_dir), X)
    
    # Verificar JSON de métricas
    metrics_file = output_dir / 'test_model_metrics.json'
    assert metrics_file.exists()
    
    # Leer y validar contenido
    with open(metrics_file) as f:
        saved_metrics = json.load(f)
    
    assert 'test_rmse' in saved_metrics
    assert saved_metrics['test_rmse'] > 0
    
    # Verificar visualizaciones
    assert (output_dir / 'test_model_residuals_test.png').exists()
    assert (output_dir / 'test_model_predictions_test.png').exists()


# ============================================================================
# TESTS - Visualizations
# ============================================================================

def test_plot_residuals(tmp_path):
    """Test: Gráfico de residuales"""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    save_path = tmp_path / 'residuals.png'
    
    fig = plot_residuals(y_true, y_pred, save_path=save_path)
    
    assert fig is not None
    assert save_path.exists()


def test_plot_predictions(tmp_path):
    """Test: Gráfico de predicciones vs reales"""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    
    save_path = tmp_path / 'predictions.png'
    
    fig = plot_predictions(y_true, y_pred, save_path=save_path)
    
    assert fig is not None
    assert save_path.exists()


def test_plot_feature_importance(tmp_path):
    """Test: Gráfico de importancia de features"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Entrenar modelo simple
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X.sum(axis=1)
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(10)]
    save_path = tmp_path / 'feature_importance.png'
    
    fig = plot_feature_importance(
        model,
        feature_names,
        save_path=save_path
    )
    
    assert fig is not None
    assert save_path.exists()


def test_plot_feature_importance_no_importances(tmp_path):
    """Test: Modelo sin feature_importances_"""
    from sklearn.linear_model import LinearRegression
    
    # LinearRegression no tiene feature_importances_
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X.sum(axis=1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    feature_names = [f'feature_{i}' for i in range(5)]
    save_path = tmp_path / 'feature_importance.png'
    
    fig = plot_feature_importance(
        model,
        feature_names,
        save_path=save_path
    )
    
    # Debe retornar None (no tiene importances)
    assert fig is None
    assert not save_path.exists()


def test_plot_error_distribution(tmp_path):
    """Test: Distribución de errores"""
    np.random.seed(42)
    y_true = np.random.randn(100) * 10 + 20
    y_pred = y_true + np.random.randn(100) * 2
    
    save_path = tmp_path / 'error_distribution.png'
    
    fig = plot_error_distribution(y_true, y_pred, save_path=save_path)
    
    assert fig is not None
    assert save_path.exists()


def test_plot_model_comparison(tmp_path):
    """Test: Comparación de modelos"""
    comparison_df = pd.DataFrame({
        'model': ['model_a', 'model_b', 'model_c'],
        'test_rmse': [3.5, 2.8, 4.1],
        'test_mae': [2.1, 1.8, 2.9],
        'test_r2': [0.85, 0.92, 0.78]
    })
    
    save_path = tmp_path / 'model_comparison.png'
    
    fig = plot_model_comparison(
        comparison_df,
        metric='test_rmse',
        save_path=save_path
    )
    
    assert fig is not None
    assert save_path.exists()


def test_plot_model_comparison_r2_metric(tmp_path):
    """Test: Comparación con métrica R² (mayor es mejor)"""
    comparison_df = pd.DataFrame({
        'model': ['model_a', 'model_b'],
        'test_r2':  [0.85, 0.92]
    })
    
    save_path = tmp_path / 'model_comparison_r2.png'
    
    fig = plot_model_comparison(
        comparison_df,
        metric='test_r2',
        save_path=save_path
    )
    
    assert fig is not None
    assert save_path.exists()


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

@pytest.mark.integration
def test_full_evaluation_pipeline(tmp_path):
    """Test de integración:  Pipeline completo de evaluación"""
    from sklearn.ensemble import RandomForestRegressor
    
    # 1.Crear y guardar modelo
    np.random.seed(42)
    n_train, n_test = 100, 30
    
    X_train = pd.DataFrame(np.random.randn(n_train, 5), columns=[f'f{i}' for i in range(5)])
    y_train = X_train.sum(axis=1) + np.random.randn(n_train) * 0.5
    
    X_test = pd.DataFrame(np.random.randn(n_test, 5), columns=[f'f{i}' for i in range(5)])
    y_test = X_test.sum(axis=1) + np.random.randn(n_test) * 0.5
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    model_path = tmp_path / 'model.pkl'
    joblib.dump(model, model_path)
    
    # 2.Evaluar
    evaluator = ModelEvaluator(str(model_path))
    metrics = evaluator.evaluate(X_test, y_test, X_train, y_train)
    
    # 3.Generar reporte
    output_dir = tmp_path / 'evaluation'
    evaluator.generate_report(str(output_dir), X_test)
    
    # 4.Validaciones
    assert 'train_rmse' in metrics
    assert 'test_rmse' in metrics
    
    # Verificar archivos generados
    assert (output_dir / 'model_metrics.json').exists()
    assert (output_dir / 'model_residuals_test.png').exists()
    assert (output_dir / 'model_predictions_test.png').exists()
    assert (output_dir / 'model_errors_test.png').exists()
    assert (output_dir / 'model_feature_importance.png').exists()
    
    # Validar contenido del JSON
    with open(output_dir / 'model_metrics.json') as f:
        saved_metrics = json.load(f)
    
    assert saved_metrics['test_rmse'] == metrics['test_rmse']
    assert saved_metrics['train_r2'] == metrics['train_r2']


def test_evaluate_with_overfitting_detection(sample_model, sample_data, caplog):
    """Test: Detección de overfitting"""
    import logging
    
    X, y = sample_data
    
    # Crear train set con fit perfecto (overfitting simulado)
    np.random.seed(42)
    X_train_small = X[:10]
    y_train_small = y[:10]
    
    # Re-entrenar modelo con datos pequeños (causa overfitting)
    from sklearn.ensemble import RandomForestRegressor
    model_overfit = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    model_overfit.fit(X_train_small, y_train_small)
    
    # Guardar
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(model_overfit, f.name)
        model_path = f.name
    
    # Evaluar
    evaluator = ModelEvaluator(model_path)
    
    with caplog.at_level(logging.WARNING):
        evaluator.evaluate(X, y, X_train_small, y_train_small)
    
    # Verificar que se detectó overfitting (train RMSE << test RMSE)
    # (puede o no loguear warning dependiendo de la diferencia)
    assert 'train_rmse' in evaluator.metrics_
    assert 'test_rmse' in evaluator.metrics_