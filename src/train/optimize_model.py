"""
Optimización de hiperparámetros con Optuna
"""

import optuna
import mlflow
import numpy as np
from sklearn.model_selection import cross_val_score
from .train_model import load_model_class
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelOptimizer:
    """Optimizador de hiperparámetros con Optuna + MLflow"""
    
    def __init__(self, model_name:  str, config: dict):
        self.model_name = model_name
        self.config = config
        self.study = None
        self.best_model = None
    
    def _get_param_space(self, trial):
        """Define espacio de búsqueda según modelo"""
        
        if self.model_name == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
        
        elif self.model_name == 'lightgbm': 
            return {
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate':  trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda':  trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        
        # ...más modelos según necesites
        
        else:
            raise ValueError(f"Optimización no implementada para {self.model_name}")
    
    def _objective(self, trial, X, y):
        """Función objetivo para Optuna"""
        
        # Obtener espacio de búsqueda
        params = self._get_param_space(trial)
        
        # Cargar clase del modelo
        model_config = self.config['models'][self.model_name]
        ModelClass = load_model_class(model_config['module'], model_config['class'])
        
        # Crear modelo
        model = ModelClass(**params)
        
        # Cross-validation
        cv_config = self.config['validation']
        scores = cross_val_score(
            model, X, y,
            cv=cv_config['cv_folds'],
            scoring=cv_config['cv_scoring'],
            n_jobs=-1
        )
        
        rmse = -scores.mean()
        
        # Log en MLflow (opcional)
        with mlflow.start_run(nested=True, run_name=f"optuna_trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metric('cv_rmse', rmse)
        
        return rmse
    
    def optimize(self, X_train, y_train, n_trials=100):
        """Ejecuta optimización Optuna"""
        
        logger.info("="*60)
        logger.info(f"OPTIMIZACIÓN DE HIPERPARÁMETROS:  {self.model_name}")
        logger.info("="*60)
        logger.info(f"Trials: {n_trials}")
        
        # Crear estudio Optuna
        self.study = optuna.create_study(
            direction='minimize',
            study_name=f'{self.model_name}_optimization'
        )
        
        # Optimizar
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"\n✅ Optimización completada")
        logger.info(f"Mejores parámetros: {self.study.best_params}")
        logger.info(f"Mejor RMSE (CV): {self.study.best_value:.4f}")
        
        return self.study.best_params
    
    def train_best_model(self, X_train, y_train, X_test, y_test):
        """Entrena modelo final con mejores parámetros"""
        
        best_params = self.study.best_params
        
        # Cargar clase
        model_config = self.config['models'][self.model_name]
        ModelClass = load_model_class(model_config['module'], model_config['class'])
        
        # Entrenar
        self.best_model = ModelClass(**best_params)
        self.best_model.fit(X_train, y_train)
        
        # Evaluar
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred_train = self.best_model.predict(X_train)
        y_pred_test = self.best_model.predict(X_test)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"\n📊 Métricas del modelo optimizado:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return self.best_model, metrics
    
    def save_best_model(self, output_path):
        """Guarda modelo optimizado"""
        import joblib
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_model, output_path)
        logger.info(f"✅ Modelo optimizado guardado:  {output_path}")