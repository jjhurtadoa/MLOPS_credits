"""
Configuración de la API
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # API
    app_name: str = "Boston Housing Price Prediction API"
    app_version: str = "1.0.0"
    app_description: str = "API para predecir precios de viviendas en Boston usando ML"
    
    # Modelo
    model_path: str = "artifacts/models/best_model.pkl"
    model_name: str = "best_model"
    
    # Preprocessors
    preprocessor_dir: str = "artifacts/preprocessors"
    data_preprocessor_path: str = "artifacts/preprocessors/data_preprocessor.pkl"
    feature_engineer_path: str = "artifacts/preprocessors/feature_engineer.pkl"
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # CORS
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()