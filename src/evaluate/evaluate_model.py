"""
Evaluador de modelos con métricas y visualizaciones

Clase principal:  ModelEvaluator
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import json
from datetime import datetime

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

from .visualizations import (
    plot_residuals,
    plot_predictions,
    plot_feature_importance,
    plot_error_distribution,
    plot_model_comparison
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator: 
    """
    Evaluador de modelos con métricas detalladas y visualizaciones
    
    Attributes:
        model:  Modelo cargado
        model_name: Nombre del modelo
        metrics_:  Métricas calculadas
        
    Example:
        >>> evaluator = ModelEvaluator('artifacts/models/best_model.pkl')
        >>> evaluator.evaluate(X_test, y_test)
        >>> evaluator.generate_report('artifacts/evaluation/')
    """
    
    def __init__(self, model_path: str, model_name: Optional[str] = None):
        """
        Args:
            model_path:  Ruta al modelo .pkl
            model_name:  Nombre del modelo (opcional)
        """
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.stem
        self.model = self._load_model()
        self.metrics_ = {}
        self.predictions_ = {}
        
        logger.info(f"✓ ModelEvaluator inicializado:  {self.model_name}")
    
    def _load_model(self) -> Any:
        """Carga modelo desde .pkl"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        logger.info(f"📂 Cargando modelo: {self.model_path}")
        model = joblib.load(self.model_path)
        logger.info(f"✓ Modelo cargado:  {type(model).__name__}")
        
        return model
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Calcula métricas de regresión
        
        Args: 
            y_true: Valores reales
            y_pred:  Predicciones
            prefix:  Prefijo para nombres de métricas
            
        Returns: 
            Dict con métricas
        """
        metrics = {
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}mae': mean_absolute_error(y_true, y_pred),
            f'{prefix}r2':  r2_score(y_true, y_pred),
            f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            f'{prefix}max_error': np.abs(y_true - y_pred).max(),
            f'{prefix}median_error': np.median(np.abs(y_true - y_pred))
        }
        
        return metrics
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Evalúa modelo en test (y opcionalmente train)
        
        Args: 
            X_test: Features de test
            y_test:  Target de test
            X_train:  Features de train (opcional)
            y_train: Target de train (opcional)
            
        Returns:
            Dict con todas las métricas
        """
        logger.info("\n" + "="*60)
        logger.info(f"📊 EVALUANDO MODELO: {self.model_name.upper()}")
        logger.info("="*60)
        
        # Predicciones en test
        y_test_pred = self.model.predict(X_test)
        self.predictions_['test'] = {
            'y_true': y_test.values,
            'y_pred':  y_test_pred
        }
        
        # Métricas de test
        test_metrics = self.calculate_metrics(y_test, y_test_pred, prefix='test_')
        self.metrics_.update(test_metrics)
        
        logger.info("\n📈 Métricas de TEST:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        # Predicciones en train (opcional)
        if X_train is not None and y_train is not None: 
            y_train_pred = self.model.predict(X_train)
            self.predictions_['train'] = {
                'y_true': y_train.values,
                'y_pred':  y_train_pred
            }
            
            train_metrics = self.calculate_metrics(y_train, y_train_pred, prefix='train_')
            self.metrics_.update(train_metrics)
            
            logger.info("\n📈 Métricas de TRAIN:")
            for metric, value in train_metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
            
            # Detectar overfitting
            rmse_diff = train_metrics['train_rmse'] - test_metrics['test_rmse']
            if rmse_diff < -1.0:
                logger.warning(f"⚠️ OVERFITTING detectado")
                logger.warning(f"   Train RMSE: {train_metrics['train_rmse']:.4f}")
                logger.warning(f"   Test RMSE:  {test_metrics['test_rmse']:.4f}")
                logger.warning(f"   Diferencia: {rmse_diff:.4f}")
        
        return self.metrics_
    
    def generate_visualizations(
        self,
        output_dir: str,
        X_test: Optional[pd.DataFrame] = None
    ):
        """
        Genera todas las visualizaciones
        
        Args:
            output_dir:  Directorio de salida
            X_test: Features (para feature names)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*60)
        logger.info("📊 GENERANDO VISUALIZACIONES")
        logger.info("="*60)
        
        # 1.Residual plot (test)
        if 'test' in self.predictions_:
            plot_residuals(
                self.predictions_['test']['y_true'],
                self.predictions_['test']['y_pred'],
                title=f"{self.model_name} - Test Set",
                save_path=output_path / f'{self.model_name}_residuals_test.png'
            )
        
        # 2.Predictions plot (test)
        if 'test' in self.predictions_: 
            plot_predictions(
                self.predictions_['test']['y_true'],
                self.predictions_['test']['y_pred'],
                title=f"{self.model_name} - Predicted vs Actual (Test)",
                save_path=output_path / f'{self.model_name}_predictions_test.png'
            )
        
        # 3.Error distribution
        if 'test' in self.predictions_:
            plot_error_distribution(
                self.predictions_['test']['y_true'],
                self.predictions_['test']['y_pred'],
                title=f"{self.model_name} - Test Set",
                save_path=output_path / f'{self.model_name}_errors_test.png'
            )
        
        # 4.Feature importance
        if hasattr(self.model, 'feature_importances_') and X_test is not None:
            plot_feature_importance(
                self.model,
                feature_names=X_test.columns.tolist(),
                title=f"{self.model_name} - Feature Importance",
                save_path=output_path / f'{self.model_name}_feature_importance.png'
            )
        
        logger.info(f"\n✓ Visualizaciones guardadas en: {output_path}")
    
    def generate_report(
        self,
        output_dir: str,
        X_test: Optional[pd.DataFrame] = None
    ):
        """
        Genera reporte completo (JSON + visualizaciones)
        
        Args:
            output_dir:  Directorio de salida
            X_test: Features (para visualizaciones)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas en JSON
        metrics_file = output_path / f'{self.model_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_, f, indent=2)
        
        logger.info(f"\n✓ Métricas guardadas:  {metrics_file}")
        
        # Generar visualizaciones
        self.generate_visualizations(output_dir, X_test)
        
        logger.info("\n" + "="*60)
        logger.info("✅ REPORTE COMPLETADO")
        logger.info("="*60)