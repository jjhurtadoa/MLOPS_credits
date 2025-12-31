"""
Módulo de preprocesamiento de datos

Implementa todas las estrategias definidas en preprocessing_config.yaml:
- Manejo de valores faltantes (mean, median, most_frequent)
- Manejo de outliers (log transform, IQR capping, Winsorization)
- Reducción de correlación
- Escalado de features

Todas las clases son compatibles con sklearn Pipeline y persistibles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Literal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats.mstats import winsorize
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# MISSING VALUE HANDLER
# ============================================================================

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Maneja valores faltantes según estrategias configuradas
    
    Estrategias soportadas:
    - mean:  Imputar con media
    - median:  Imputar con mediana
    - most_frequent: Imputar con moda
    - no_action: No hacer nada
    
    Attributes:
        config (dict): Configuración de estrategias por columna
        imputers_ (dict): Imputers entrenados por estrategia
        

    """
    
    def __init__(self, config:  Dict[str, List[str]]):
        """
        Args:
            config:  Diccionario con estrategia -> lista de columnas
        """
        self.config = config
        self.imputers_ = {}
        self.feature_names_in_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Entrena los imputers en datos de entrenamiento
        
        Args:
            X: DataFrame de features
            y: Target (no usado, para compatibilidad sklearn)
            
        Returns: 
            self: Objeto fitted
        """
        logger.info("="*60)
        logger.info("FITTING MISSING VALUE HANDLER")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Verificar valores faltantes
        missing_counts = X.isnull().sum()
        total_missing = missing_counts.sum()
        
        logger.info(f"Total valores faltantes: {total_missing}")
        if total_missing > 0:
            logger.info(f"Columnas con NAs:\n{missing_counts[missing_counts > 0]}")
        
        # Entrenar imputers por estrategia
        for strategy, columns in self.config.items():
            if strategy == 'no_action' or not columns:
                continue
                
            # Validar que columnas existen
            valid_columns = [col for col in columns if col in X.columns]
            if not valid_columns: 
                logger.warning(f"Estrategia '{strategy}':  ninguna columna válida")
                continue
            
            # Crear y entrenar imputer
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(X[valid_columns])
            
            self.imputers_[strategy] = {
                'imputer': imputer,
                'columns': valid_columns
            }
            
            logger.info(f"✓ Estrategia '{strategy}':  {len(valid_columns)} columnas")
            logger.debug(f"  Columnas: {valid_columns}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica imputación a datos
        
        Args:
            X: DataFrame de features
            
        Returns:
            DataFrame con valores imputados
        """
        X_transformed = X.copy()
        
        for strategy, data in self.imputers_.items():
            imputer = data['imputer']
            columns = data['columns']
            
            # Aplicar imputación
            X_transformed[columns] = imputer.transform(X_transformed[columns])
            
            logger.debug(f"Aplicada imputación '{strategy}' a {len(columns)} columnas")
        
        # Verificar que no quedan NAs (excepto en no_action)
        no_action_cols = self.config.get('no_action', [])
        remaining_nas = X_transformed.drop(columns=no_action_cols, errors='ignore').isnull().sum().sum()
        
        if remaining_nas > 0:
            logger.warning(f"⚠️  Quedan {remaining_nas} valores faltantes después de imputación")
        else:
            logger.info("✓ Todos los valores faltantes imputados correctamente")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Compatibilidad con sklearn >= 1.0"""
        return self.feature_names_in_


# ============================================================================
# OUTLIER HANDLER
# ============================================================================

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Maneja outliers según estrategias configuradas
    
    Estrategias soportadas: 
    - log_transform:  Aplicar log(x + 1)
    - capping_iqr: Cap usando IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    - capping_winsor:  Winsorization (caps extremos)
    - remove:  Marca filas para eliminar (NO recomendado en transform)
    - no_action: No hacer nada
    
    Attributes: 
        config (dict): Configuración de estrategias
        statistics_ (dict): Estadísticas calculadas en fit (Q1, Q3, etc.)
    """
    
    def __init__(
        self, 
        config:  Dict[str, List[str]],
        iqr_factor: float = 1.5,
        winsor_limits: tuple = (0.05, 0.05)
    ):
        """
        Args:
            config: Diccionario estrategia -> columnas
            iqr_factor: Factor para capping IQR (default: 1.5)
            winsor_limits: Límites para Winsorization (default: 5% cada lado)
        """
        self.config = config
        self.iqr_factor = iqr_factor
        self.winsor_limits = winsor_limits
        self.statistics_ = {}
        self.feature_names_in_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calcula estadísticas necesarias para manejo de outliers"""
        logger.info("="*60)
        logger.info("FITTING OUTLIER HANDLER")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Calcular estadísticas para capping_iqr
        iqr_columns = self.config.get('capping_iqr', [])
        if iqr_columns:
            for col in iqr_columns: 
                if col not in X.columns:
                    continue
                    
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.iqr_factor * IQR
                upper_bound = Q3 + self.iqr_factor * IQR
                
                self.statistics_[col] = {
                    'strategy': 'capping_iqr',
                    'lower':  lower_bound,
                    'upper': upper_bound,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                }
                
                n_outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
                logger.info(f"✓ {col}: IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}], "
                          f"{n_outliers} outliers ({n_outliers/len(X)*100:.1f}%)")
        
        # Log para otras estrategias
        log_cols = self.config.get('log_transform', [])
        if log_cols:
            logger.info(f"✓ Log transform: {len(log_cols)} columnas")
            
        winsor_cols = self.config.get('capping_winsor', [])
        if winsor_cols: 
            logger.info(f"✓ Winsorization: {len(winsor_cols)} columnas (limits={self.winsor_limits})")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica manejo de outliers"""
        X_transformed = X.copy()
        
        # 1. Log Transform
        log_columns = self.config.get('log_transform', [])
        for col in log_columns:
            if col in X_transformed.columns:
                # Verificar valores negativos
                if (X_transformed[col] < 0).any():
                    logger.warning(f"⚠️  {col} tiene valores negativos, aplicando log(|x| + 1)")
                    X_transformed[col] = np.log1p(np.abs(X_transformed[col]))
                else:
                    X_transformed[col] = np.log1p(X_transformed[col])
                logger.debug(f"Aplicado log transform a {col}")
        
        # 2. IQR Capping
        for col, stats in self.statistics_.items():
            if stats['strategy'] == 'capping_iqr' and col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].clip(
                    lower=stats['lower'],
                    upper=stats['upper']
                )
                logger.debug(f"Aplicado IQR capping a {col}")
        
        # 3. Winsorization
        winsor_columns = self.config.get('capping_winsor', [])
        for col in winsor_columns: 
            if col in X_transformed.columns:
                X_transformed[col] = winsorize(
                    X_transformed[col],
                    limits=self.winsor_limits
                )
                logger.debug(f"Aplicado Winsorization a {col}")
        
        logger.info("✓ Outlier handling aplicado correctamente")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Compatibilidad sklearn"""
        return self.feature_names_in_


# ============================================================================
# CORRELATION REDUCER
# ============================================================================

class CorrelationReducer(BaseEstimator, TransformerMixin):
    """
    Elimina features altamente correlacionadas
    
    Attributes:
        config (dict): Configuración de columnas a eliminar/mantener
        columns_to_drop_ (list): Columnas a eliminar (fitted)
    """
    
    def __init__(self, config: Dict[str, List[str]]):
        """
        Args: 
            config: Dict con 'remove' y 'no_action'
        """
        self.config = config
        self.columns_to_drop_ = []
        self.feature_names_in_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Determina qué columnas eliminar"""
        logger.info("="*60)
        logger.info("FITTING CORRELATION REDUCER")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Columnas a eliminar según config
        remove_cols = self.config.get('remove', [])
        self.columns_to_drop_ = [col for col in remove_cols if col in X.columns]
        
        if self.columns_to_drop_: 
            logger.info(f"✓ Columnas a eliminar por correlación: {self.columns_to_drop_}")
            
            # Mostrar correlaciones (si están disponibles)
            if len(self.columns_to_drop_) > 0:
                for col in self.columns_to_drop_:
                    corr_with_target = X.corrwith(X[col]).sort_values(ascending=False)
                    logger.debug(f"\n  Correlaciones de {col}:\n{corr_with_target.head()}")
        else:
            logger.info("✓ No se eliminarán columnas por correlación")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas correlacionadas"""
        X_transformed = X.drop(columns=self.columns_to_drop_, errors='ignore')
        
        if self.columns_to_drop_:
            logger.info(f"✓ Eliminadas {len(self.columns_to_drop_)} columnas correlacionadas")
            logger.debug(f"  Shape:  {X.shape} -> {X_transformed.shape}")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Retorna nombres de features después de eliminar correlacionadas"""
        if input_features is None:
            input_features = self.feature_names_in_
        return [col for col in input_features if col not in self.columns_to_drop_]


# ============================================================================
# FEATURE SCALER
# ============================================================================

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Escala features usando estrategia configurada
    
    Soporta:
    - RobustScaler (resistente a outliers)
    - StandardScaler (z-score normalization)
    - MinMaxScaler (0-1 normalization)
    
    Attributes:
        scaler_type (str): Tipo de scaler
        scaler_ (object): Scaler sklearn fitted
    """
    
    def __init__(
        self, 
        scaler_type:  Literal['RobustScaler', 'StandardScaler', 'MinMaxScaler'] = 'RobustScaler'
    ):
        """
        Args:
            scaler_type: Tipo de scaler a usar
        """
        self.scaler_type = scaler_type
        self.scaler_ = None
        self.feature_names_in_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Entrena el scaler"""
        logger.info("="*60)
        logger.info(f"FITTING {self.scaler_type.upper()}")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Crear scaler según tipo
        if self.scaler_type == 'RobustScaler':
            self.scaler_ = RobustScaler()
        elif self.scaler_type == 'StandardScaler':
            self.scaler_ = StandardScaler()
        elif self.scaler_type == 'MinMaxScaler': 
            from sklearn.preprocessing import MinMaxScaler
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError(f"Scaler type '{self.scaler_type}' no soportado")
        
        # Fit
        self.scaler_.fit(X)
        
        logger.info(f"✓ {self.scaler_type} fitted en {X.shape[1]} features")
        
        # Log estadísticas (solo para algunos scalers)
        if hasattr(self.scaler_, 'center_'):
            logger.debug(f"  Center (median): {self.scaler_.center_[: 3]}...")
        if hasattr(self.scaler_, 'scale_'):
            logger.debug(f"  Scale (IQR): {self.scaler_.scale_[:3]}...")
        
        return self
    
    def transform(self, X:  pd.DataFrame) -> pd.DataFrame:
        """Aplica scaling"""
        X_scaled = self.scaler_.transform(X)
        
        # Convertir de vuelta a DataFrame
        X_transformed = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )
        
        logger.info(f"✓ Scaling aplicado con {self.scaler_type}")
        logger.debug(f"  Shape: {X_transformed.shape}")
        logger.debug(f"  Range: [{X_transformed.min().min():.2f}, {X_transformed.max().max():.2f}]")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Compatibilidad sklearn"""
        return self.feature_names_in_


# ============================================================================
# DATA PREPROCESSOR (Pipeline Completo)
# ============================================================================

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline completo de preprocesamiento
    
    Aplica en orden:
    1. Manejo de valores faltantes
    2. Manejo de outliers
    3. Reducción de correlación
    4. Escalado de features
    
    Attributes:
        config (dict): Configuración completa de preprocessing_config.yaml
        missing_handler_ (MissingValueHandler): Handler de NAs fitted
        outlier_handler_ (OutlierHandler): Handler de outliers fitted
        correlation_reducer_ (CorrelationReducer): Reducer fitted
        scaler_ (FeatureScaler): Scaler fitted
        
    Example:
        >>> from src.utils.config_loader import load_config
        >>> config = load_config('src/configs/preprocessing_config.yaml')
        >>> preprocessor = DataPreprocessor(config['feature_processing'])
        >>> X_train_processed = preprocessor.fit_transform(X_train)
        >>> X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(self, config: Dict):
        """
        Args: 
            config: Sección 'feature_processing' del config YAML
        """
        self.config = config
        self.missing_handler_ = None
        self.outlier_handler_ = None
        self.correlation_reducer_ = None
        self.scaler_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Entrena todos los transformers
        
        Args: 
            X: Features de entrenamiento
            y:  Target (opcional, no usado)
            
        Returns:
            self:  Preprocessor fitted
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 FITTING COMPLETE DATA PREPROCESSOR")
        logger.info("="*60)
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Input features: {X.columns.tolist()}")
        
        self.feature_names_in_ = X.columns.tolist()
        X_temp = X.copy()
        
        # 1. Missing Values
        if 'handling_na' in self.config:
            self.missing_handler_ = MissingValueHandler(self.config['handling_na'])
            X_temp = self.missing_handler_.fit_transform(X_temp)
        
        # 2. Outliers
        if 'handling_outliers' in self.config:
            self.outlier_handler_ = OutlierHandler(self.config['handling_outliers'])
            X_temp = self.outlier_handler_.fit_transform(X_temp)
        
        # 3. Correlation
        if 'handling_correlated_features' in self.config:
            self.correlation_reducer_ = CorrelationReducer(
                self.config['handling_correlated_features']
            )
            X_temp = self.correlation_reducer_.fit_transform(X_temp)
        
        # 4. Scaling
        if 'scaler' in self.config:
            self.scaler_ = FeatureScaler(self.config['scaler'])
            X_temp = self.scaler_.fit_transform(X_temp)
        
        self.feature_names_out_ = X_temp.columns.tolist()
        
        logger.info("\n" + "="*60)
        logger.info("✅ PREPROCESSING FITTED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Output shape: {X_temp.shape}")
        logger.info(f"Output features: {self.feature_names_out_}")
        logger.info(f"Features removed: {set(self.feature_names_in_) - set(self.feature_names_out_)}")
        
        return self
    
    def transform(self, X:  pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones
        
        Args:
            X: Features a transformar
            
        Returns: 
            DataFrame transformado
        """
        logger.info(f"\n🔄 Transformando datos:  {X.shape}")
        
        X_transformed = X.copy()
        
        # Aplicar transformaciones en orden
        if self.missing_handler_:
            X_transformed = self.missing_handler_.transform(X_transformed)
            
        if self.outlier_handler_:
            X_transformed = self.outlier_handler_.transform(X_transformed)
            
        if self.correlation_reducer_:
            X_transformed = self.correlation_reducer_.transform(X_transformed)
            
        if self.scaler_:
            X_transformed = self.scaler_.transform(X_transformed)
        
        logger.info(f"✓ Transformación completada:  {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit y transform en un solo paso"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Retorna nombres de features de salida"""
        return self.feature_names_out_
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Guarda preprocessor en disco
        
        Args:
            filepath: Ruta donde guardar (.  pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"💾 Preprocessor guardado en: {filepath}")
    
    @staticmethod
    def load(filepath:  Union[str, Path]) -> 'DataPreprocessor':
        """
        Carga preprocessor desde disco
        
        Args: 
            filepath: Ruta del archivo .  pkl
            
        Returns: 
            DataPreprocessor cargado
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"📂 Preprocessor cargado desde: {filepath}")
        return preprocessor