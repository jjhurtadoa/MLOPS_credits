"""
Utilidades para la API
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class PreprocessorLoader:
    """
    Singleton para cargar y cachear preprocessors
    """
    
    _instance = None
    _data_preprocessor = None
    _feature_engineer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PreprocessorLoader, cls).__new__(cls)
        return cls._instance
    
    def load_preprocessors(self, data_preprocessor_path: str, feature_engineer_path: str):
        """
        Carga preprocessors desde disco
        
        Args:
            data_preprocessor_path: Ruta al data_preprocessor.pkl
            feature_engineer_path: Ruta al feature_engineer.pkl
        """
        if self._data_preprocessor is None or self._feature_engineer is None:
            # Cargar data preprocessor
            dp_file = Path(data_preprocessor_path)
            if dp_file.exists():
                logger.info(f"Cargando data preprocessor desde: {data_preprocessor_path}")
                self._data_preprocessor = joblib.load(dp_file)
                logger.info("✓ Data preprocessor cargado")
            else:
                logger. warning(f"⚠️ Data preprocessor no encontrado:  {data_preprocessor_path}")
            
            # Cargar feature engineer
            fe_file = Path(feature_engineer_path)
            if fe_file.exists():
                logger.info(f"Cargando feature engineer desde: {feature_engineer_path}")
                self._feature_engineer = joblib.load(fe_file)
                logger.info("✓ Feature engineer cargado")
            else:
                logger.warning(f"⚠️ Feature engineer no encontrado: {feature_engineer_path}")
        
        return self._data_preprocessor, self._feature_engineer
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica preprocessing completo a los datos
        
        Args: 
            X: DataFrame con features originales (13 columnas)
            
        Returns: 
            DataFrame con features procesadas
        """
        if self._data_preprocessor is None or self._feature_engineer is None:
            raise ValueError("Preprocessors no están cargados")
        
        logger.debug(f"Input features:  {X.columns.tolist()}")
        
        # 1. Data preprocessing (limpieza, outliers, scaling, etc.)
        X_preprocessed = self._data_preprocessor.transform(X)
        logger.debug(f"Después de data_preprocessor: {X_preprocessed.columns.tolist()}")
        
        # 2. Feature engineering
        X_engineered = self._feature_engineer.transform(X_preprocessed)
        logger.debug(f"Después de feature_engineer: {X_engineered.columns.tolist()}")
        
        return X_engineered
    
    @property
    def is_loaded(self) -> bool:
        """Verifica si los preprocessors están cargados"""
        return self._data_preprocessor is not None and self._feature_engineer is not None


class ModelLoader:
    """
    Singleton para cargar y cachear el modelo
    """
    
    _instance = None
    _model = None
    _model_name = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str) -> Any:
        """Carga modelo desde disco"""
        if self._model is None: 
            model_file = Path(model_path)
            
            if not model_file.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
            
            logger.info(f"Cargando modelo desde: {model_path}")
            self._model = joblib.load(model_file)
            self._model_name = model_file.stem
            logger.info(f"✓ Modelo cargado: {type(self._model).__name__}")
        
        return self._model
    
    def get_model_name(self) -> str:
        """Retorna nombre del modelo"""
        return self._model_name or "unknown"
    
    @property
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado"""
        return self._model is not None


# Instancias globales
model_loader = ModelLoader()
preprocessor_loader = PreprocessorLoader()