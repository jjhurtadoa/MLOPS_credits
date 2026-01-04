"""
Módulo de entrenamiento de modelos con import dinámico
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import yaml
from datetime import datetime
import importlib

# ML core
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# MLflow
import mlflow
import mlflow.sklearn

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# DYNAMIC MODEL LOADING
# ============================================================================

def load_model_class(module_name: str, class_name: str):
    """
    Importa dinámicamente una clase de modelo
    
    Args:
        module_name:   Nombre del módulo (ej: 'xgboost', 'sklearn.ensemble')
        class_name:  Nombre de la clase (ej:   'XGBRegressor')
        
    Returns:
        Clase del modelo
        
    Raises:
        ImportError: Si el módulo no está disponible
        AttributeError: Si la clase no existe en el módulo
        
    Example:
        >>> load_model_class('xgboost', 'XGBRegressor')
        <class 'xgboost.sklearn.XGBRegressor'>
    """
    try:
        # Importar módulo dinámicamente
        module = importlib.import_module(module_name)
        
        # Obtener clase del módulo
        model_class = getattr(module, class_name)
        
        logger.debug(f"✓ Loaded {class_name} from {module_name}")
        
        return model_class
        
    except ImportError as e: 
        raise ImportError(
            f"❌ No se pudo importar módulo '{module_name}'.\n"
            f"   Error: {e}\n"
            f"   💡 Instala con:   pip install {module_name}"
        )
    
    except AttributeError as e: 
        raise AttributeError(
            f"❌ Clase '{class_name}' no encontrada en '{module_name}'.\n"
            f"   Error: {e}\n"
            f"   💡 Verifica que el nombre de la clase sea correcto"
        )


# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainer:
    """
    Entrenador de modelos con MLflow tracking
    """
    
    def __init__(self, config:  Dict[str, Any]):
        """
        Args:
            config:  Configuración completa de model_config.yaml
        """
        self.config = config
        self.models_ = {}
        self.metrics_ = {}
        self.cv_scores_ = {}
        
        # Configurar MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Configura MLflow tracking con soporte para Windows"""
        from pathlib import Path
        import platform
        
        mlflow_config = self.config['training']['mlflow']
        
        # Tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', 'artifacts/mlruns')
        
        
        if not tracking_uri.startswith(('http://', 'https://', 'databricks', 'file://')):
            tracking_path = Path(tracking_uri).resolve()
            tracking_path.mkdir(parents=True, exist_ok=True)
            
            if platform.system() == 'Windows':
                tracking_uri = tracking_path.as_uri()
            else:
                tracking_uri = f"file://{tracking_path}"
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Experiment
        experiment_name = mlflow_config.get('experiment_name', 'boston_housing')
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"✓ MLflow configurado:")
        logger.info(f"  - Tracking URI: {tracking_uri}")
        logger.info(f"  - Experiment: {experiment_name}")
    
    def load_data(
        self, 
        data_dir: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Carga datos procesados"""
        data_dir = data_dir or self.config['training']['data_dir']
        target_column = target_column or self.config['training']['target_column']
        
        data_path = Path(data_dir)
        train_path = data_path / 'train_processed.csv'
        test_path = data_path / 'test_processed.csv'
        
        logger.info(f"📂 Cargando datos desde:   {data_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        logger.info(f"✓ Datos cargados:")
        logger.info(f"  - Train:   {X_train.shape}")
        logger.info(f"  - Test:   {X_test.shape}")
        logger.info(f"  - Features: {X_train.shape[1]}")
        logger.info(f"  - Target:   {target_column}")
        
        return X_train, X_test, y_train, y_test
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas de regresión"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        
        return metrics
    
    def cross_validate_model(
        self,
        model:   Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """Cross-validation del modelo"""
        cv_config = self.config.get('validation', {})
        cv_folds = cv_config.get('cv_folds', 5)
        cv_scoring = cv_config.get('cv_scoring', 'neg_root_mean_squared_error')
        
        logger.info(f"  Ejecutando {cv_folds}-fold cross-validation...")
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring=cv_scoring,
            n_jobs=-1
        )
        
        if cv_scoring.startswith('neg_'):
            cv_scores = -cv_scores
        
        cv_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        
        logger.info(f"  ✓ CV RMSE:  {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        
        return cv_results
    
    def train_model(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Entrena un modelo individual con MLflow tracking
        
        Args:
            model_name:  Nombre del modelo
            model_config: Configuración del modelo (debe incluir 'module' y 'class')
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de prueba
            
        Returns:  
            Dict con modelo, métricas y metadata
        """
        logger.info("="*60)
        logger.info(f"ENTRENANDO:   {model_name.upper()}")
        logger.info("="*60)
        
        # ============================================================
        # IMPORT DINÁMICO - Lee module y class del YAML
        # ============================================================
        module_name = model_config.get('module')
        class_name = model_config.get('class')
        
        if not module_name or not class_name:
            logger.error(f"❌ Config inválido para '{model_name}': falta 'module' o 'class'")
            return None
        
        try: 
            model_class = load_model_class(module_name, class_name)
            logger.info(f"  Instanciando {class_name}...")
        except (ImportError, AttributeError) as e:
            logger.error(str(e))
            return None
        
        # Parámetros
        # copy params so we can inject library-specific options without mutating config
        params = dict(model_config.get('params', {}))
        
        # Prepare model_dir early (used by CatBoost train_dir)
        model_dir = Path(self.config['training']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # If using CatBoost, ensure it writes its train_dir inside artifacts and doesn't try to create
        # a folder in the CWD. Inject train_dir and silence logging.
        if module_name and module_name.startswith('catboost') or model_name.lower() == 'catboost':
            catboost_dir = model_dir / 'catboost_info'
            catboost_dir.mkdir(parents=True, exist_ok=True)
            params.setdefault('train_dir', str(catboost_dir))
            # prefer quiet training in batch runs
            params.setdefault('logging_level', 'Silent')

        # Iniciar MLflow run
        run_name = f"{self.config['training']['mlflow']['run_name_prefix']}_{model_name}"

        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params(params)
            mlflow.log_param('model_type', class_name)
            mlflow.log_param('model_module', module_name)
            mlflow.log_param('n_features', X_train.shape[1])
            mlflow.log_param('n_train_samples', X_train.shape[0])
            mlflow.log_param('n_test_samples', X_test.shape[0])

            # Crear modelo
            model = model_class(**params)

            # Cross-validation
            cv_results = self.cross_validate_model(model, X_train, y_train)
            for metric_name, metric_value in cv_results.items():
                mlflow.log_metric(metric_name, metric_value)

            # Entrenar
            logger.info(f"  Entrenando en train set completo...")
            model.fit(X_train, y_train)

            # Predicciones
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Métricas train
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f'train_{metric_name}', metric_value)

            # Métricas test
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f'test_{metric_name}', metric_value)

            # Log modelo
            mlflow.sklearn.log_model(
                model, 
                artifact_path="model",
                registered_model_name=f"boston_housing_{model_name}"
            )

            # Guardar modelo localmente
            # model_dir already created above
            model_path = model_dir / f'{model_name}_model.pkl'
            joblib.dump(model, model_path)

            # Metadata
            metadata = {
                'model_name': model_name,
                'model_class': class_name,
                'model_module': module_name,
                'params': params,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_results': cv_results,
                'model_path': str(model_path),
                'mlflow_run_id': mlflow.active_run().info.run_id,
                'trained_at': datetime.now().isoformat()
            }

            # Log metadata
            metadata_path = model_dir / f'{model_name}_metadata.yaml'
            # Write metadata (original behavior)
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

            mlflow.log_artifact(str(metadata_path))

            # Logging
            logger.info(f"\n📊 RESULTADOS - {model_name.upper()}")
            logger.info(f"  Train:")
            for metric, value in train_metrics.items():
                logger.info(f"    - {metric.upper()}: {value:.4f}")

            logger.info(f"  Test:")
            for metric, value in test_metrics.items():
                logger.info(f"    - {metric.upper()}: {value:.4f}")

            logger.info(f"  Cross-Validation:")
            logger.info(f"    - Mean RMSE: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")

            logger.info(f"\n💾 Modelo guardado:")
            logger.info(f"  - Local: {model_path}")
            logger.info(f"  - MLflow Run ID: {mlflow.active_run().info.run_id}")
        
        return {
            'model': model,
            'metadata': metadata
        }
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Entrena todos los modelos configurados"""
        logger.info("\n" + "="*60)
        logger.info("🚀 ENTRENANDO TODOS LOS MODELOS")
        logger.info("="*60)
        
        models_config = self.config['models']
        
        for model_name, model_config in models_config.items():
            # Verificar si está habilitado
            if not model_config.get('enabled', True):
                logger.info(f"⏭️  {model_name} deshabilitado, saltando...")
                continue
            
            # Entrenar
            try:
                result = self.train_model(
                    model_name, model_config,
                    X_train, y_train,
                    X_test, y_test
                )
                
                if result: 
                    self.models_[model_name] = result['model']
                    self.metrics_[model_name] = result['metadata']
                    
            except Exception as e:
                logger.error(f"❌ Error entrenando {model_name}: {e}", exc_info=True)
        
        logger.info("\n" + "="*60)
        logger.info("✅ ENTRENAMIENTO COMPLETADO")
        logger.info("="*60)
        logger.info(f"Modelos entrenados: {len(self.models_)}")
    
    def get_best_model(self, metric:   str = 'test_rmse') -> Tuple[str, Any, Dict]:  
        """Obtiene el mejor modelo según una métrica"""
        if not self.metrics_:
            raise ValueError("No hay modelos entrenados")
        
        model_scores = {}
        for model_name, metadata in self.metrics_.items():
            if 'test_metrics' in metadata:
                metric_key = metric.replace('test_', '')
                score = metadata['test_metrics'].get(metric_key)
            else:
                score = metadata.get(metric)
            
            if score is not None:
                model_scores[model_name] = score
        
        if not model_scores:
            raise ValueError(f"Métrica '{metric}' no encontrada")
        
        # Mejor modelo
        if 'r2' in metric:  
            best_model_name = max(model_scores, key=model_scores.get)
        else:
            best_model_name = min(model_scores, key=model_scores.get)
        
        best_model = self.models_[best_model_name]
        best_metadata = self.metrics_[best_model_name]
        
        logger.info(f"\n🏆 MEJOR MODELO:   {best_model_name.upper()}")
        logger.info(f"  {metric}:   {model_scores[best_model_name]:.4f}")
        
        return best_model_name, best_model, best_metadata
    
    def compare_models(self) -> pd.DataFrame:
        """Compara métricas de todos los modelos"""
        comparison_data = []
        
        for model_name, metadata in self.metrics_.items():
            row = {'model': model_name}
            
            if 'train_metrics' in metadata:
                for metric, value in metadata['train_metrics'].items():
                    row[f'train_{metric}'] = value
            
            if 'test_metrics' in metadata:
                for metric, value in metadata['test_metrics'].items():
                    row[f'test_{metric}'] = value
            
            if 'cv_results' in metadata:
                row['cv_rmse_mean'] = metadata['cv_results']['cv_mean']
                row['cv_rmse_std'] = metadata['cv_results']['cv_std']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        logger.info("\n" + "="*60)
        logger.info("📊 COMPARACIÓN DE MODELOS")
        logger.info("="*60)
        logger.info(f"\n{df.to_string(index=False)}")
        
        return df