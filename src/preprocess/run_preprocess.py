"""
Script ejecutable para pipeline de preprocesamiento

Flujo: 
    1.Carga datos divididos (train/test) desde data/splits/
    2.Carga configuración de preprocessing_config.yaml
    3.Aplica preprocessing (MissingValues, Outliers, Correlation, Scaling)
    4.Aplica feature engineering (opcional)
    5.Guarda datos transformados en data/interim/
    6.Guarda transformers (preprocessor, feature_engineer) en artifacts/

Uso:
    python -m src.preprocess.run_preprocess
    python -m src.preprocess.run_preprocess --config custom_config.yaml
    python -m src.preprocess.run_preprocess --no-feature-engineering
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import sys

from src.utils.logger import setup_logging, get_logger
from src.preprocess.preprocessing import DataPreprocessor
from src.preprocess.build_features import FeatureEngineer
from src.data import load_split_data

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    Carga configuración desde YAML
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns: 
        dict: Configuración cargada
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay error en el formato
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Archivo de configuración no encontrado: {config_path}")
    
    logger.info(f"📄 Cargando configuración desde: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ Configuración cargada exitosamente")
    logger.debug(f"  Secciones: {list(config.keys())}")
    
    return config


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = 'data/interim',
    target_column: str = 'MEDV'
):
    """
    Guarda datos procesados
    
    Args:
        X_train, X_test:  Features procesadas
        y_train, y_test: Target
        output_dir: Directorio de salida
        target_column: Nombre del target
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("GUARDANDO DATOS PROCESADOS")
    logger.info("="*60)
    
    # Combinar X e y
    train_df = X_train.copy()
    train_df[target_column] = y_train.values
    
    test_df = X_test.copy()
    test_df[target_column] = y_test.values
    
    # Guardar
    train_path = output_path / 'train_processed.csv'
    test_path = output_path / 'test_processed.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"✓ Guardado:  {train_path}")
    logger.info(f"  - Shape: {train_df.shape}")
    logger.info(f"  - Size: {train_path.stat().st_size / 1024:.1f} KB")
    
    logger.info(f"✓ Guardado: {test_path}")
    logger.info(f"  - Shape: {test_df.shape}")
    logger.info(f"  - Size: {test_path.stat().st_size / 1024:.1f} KB")
    
    logger.info(f"✓ Columnas finales ({len(train_df.columns)}): {list(train_df.columns)}")


def main():
    """
    Pipeline principal de preprocesamiento
    """
    parser = argparse.ArgumentParser(
        description='Pipeline de preprocesamiento de datos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/preprocessing_config.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/splits',
        help='Directorio con datos divididos (train.csv, test.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/interim',
        help='Directorio para guardar datos procesados'
    )
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='artifacts/preprocessors',
        help='Directorio para guardar transformers (.pkl)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='MEDV',
        help='Nombre de la columna target'
    )
    parser.add_argument(
        '--no-feature-engineering',
        action='store_true',
        help='Desactivar feature engineering'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Nivel de logging'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='artifacts/logs/preprocessing.log',
        help='Archivo de log (opcional)'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    import logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level, log_file=args.log_file)
    
    logger.info("\n" + "="*60)
    logger.info("🚀 PREPROCESSING PIPELINE - INICIO")
    logger.info("="*60)
    logger.info(f"Configuración: {args.config}")
    logger.info(f"Input:  {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Artifacts: {args.artifacts_dir}")
    logger.info(f"Feature Engineering: {not args.no_feature_engineering}")
    
    try:
        # ====================================================================
        # 1.CARGAR CONFIGURACIÓN
        # ====================================================================
        
        config = load_config(args.config)
        preprocessing_config = config.get('feature_processing', {})
        feature_engineering_config = config.get('feature_engineering', {})
        
        # ====================================================================
        # 2.CARGAR DATOS
        # ====================================================================
        
        X_train, X_test, y_train, y_test = load_split_data(
            data_dir=args.input_dir,
            target_column=args.target
        )
        
        logger.info(f"\n📊 Datos cargados:")
        logger.info(f"  - Train: {X_train.shape}")
        logger.info(f"  - Test:   {X_test.shape}")
        logger.info(f"  - Features: {list(X_train.columns)}")
        logger.info(f"  - Valores faltantes (train): {X_train.isnull().sum().sum()}")
        logger.info(f"  - Valores faltantes (test): {X_test.isnull().sum().sum()}")
        
        # ====================================================================
        # 3.PREPROCESSING
        # ====================================================================
        
        logger.info("\n" + "="*60)
        logger.info("FASE 1: DATA PREPROCESSING")
        logger.info("="*60)
        
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Fit en train
        preprocessor.fit(X_train)
        
        # Transform train y test
        X_train_preprocessed = preprocessor.transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        logger.info(f"\n✓ Preprocessing completado:")
        logger.info(f"  - Train:  {X_train.shape} -> {X_train_preprocessed.shape}")
        logger.info(f"  - Test:  {X_test.shape} -> {X_test_preprocessed.shape}")
        logger.info(f"  - Features eliminadas: {set(X_train.columns) - set(X_train_preprocessed.columns)}")
        logger.info(f"  - Valores faltantes restantes (train): {X_train_preprocessed.isnull().sum().sum()}")
        logger.info(f"  - Valores faltantes restantes (test): {X_test_preprocessed.isnull().sum().sum()}")
        
        # Guardar preprocessor
        artifacts_path = Path(args.artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        preprocessor_path = artifacts_path / 'data_preprocessor.pkl'
        preprocessor.save(preprocessor_path)
        logger.info(f"💾 Preprocessor guardado:  {preprocessor_path}")
        
        # ====================================================================
        # 4.FEATURE ENGINEERING (opcional)
        # ====================================================================
        
        if not args.no_feature_engineering and feature_engineering_config:
            logger.info("\n" + "="*60)
            logger.info("FASE 2: FEATURE ENGINEERING")
            logger.info("="*60)
            
            feature_engineer = FeatureEngineer(feature_engineering_config)
            
            # Fit en train preprocessed
            feature_engineer.fit(X_train_preprocessed)
            
            # Transform
            X_train_final = feature_engineer.transform(X_train_preprocessed)
            X_test_final = feature_engineer.transform(X_test_preprocessed)
            
            logger.info(f"\n✓ Feature engineering completado:")
            logger.info(f"  - Train: {X_train_preprocessed.shape} -> {X_train_final.shape}")
            logger.info(f"  - Test:  {X_test_preprocessed.shape} -> {X_test_final.shape}")
            logger.info(f"  - Features agregadas: {X_train_final.shape[1] - X_train_preprocessed.shape[1]}")
            logger.info(f"  - Nuevas features: {set(X_train_final.columns) - set(X_train_preprocessed.columns)}")
            
            # Guardar feature engineer
            feature_engineer_path = artifacts_path / 'feature_engineer.pkl'
            feature_engineer.save(feature_engineer_path)
            logger.info(f"💾 FeatureEngineer guardado:  {feature_engineer_path}")
            
        else:
            logger.info("\n⏭️  Feature engineering desactivado")
            X_train_final = X_train_preprocessed
            X_test_final = X_test_preprocessed
        
        # ====================================================================
        # 5.GUARDAR DATOS PROCESADOS
        # ====================================================================
        
        save_processed_data(
            X_train_final,
            X_test_final,
            y_train,
            y_test,
            output_dir=args.output_dir,
            target_column=args.target
        )
        
        # ====================================================================
        # 6.RESUMEN FINAL
        # ====================================================================
        
        logger.info("\n" + "="*60)
        logger.info("✅ PREPROCESSING PIPELINE - COMPLETADO")
        logger.info("="*60)
        
        logger.info(f"\n📊 Resumen de Transformación:")
        logger.info(f"  Input:")
        logger.info(f"    - Train: {X_train.shape}")
        logger.info(f"    - Test:   {X_test.shape}")
        logger.info(f"  Output:")
        logger.info(f"    - Train: {X_train_final.shape}")
        logger.info(f"    - Test:  {X_test_final.shape}")
        logger.info(f"  Cambios:")
        logger.info(f"    - Features originales: {X_train.shape[1]}")
        logger.info(f"    - Features finales: {X_train_final.shape[1]}")
        logger.info(f"    - Features agregadas: {X_train_final.shape[1] - X_train.shape[1]}")
        
        logger.info(f"\n📁 Archivos generados:")
        logger.info(f"  Datos:")
        logger.info(f"    - {args.output_dir}/train_processed.csv")
        logger.info(f"    - {args.output_dir}/test_processed.csv")
        logger.info(f"  Transformers:")
        logger.info(f"    - {artifacts_path}/data_preprocessor.pkl")
        if not args.no_feature_engineering and feature_engineering_config:
            logger.info(f"    - {artifacts_path}/feature_engineer.pkl")
        
        logger.info(f"\n🔜 Siguiente paso:")
        logger.info(f"  python -m src.train.run_train")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR:  Archivo no encontrado")
        logger.error(f"{str(e)}")
        logger.info(f"\n💡 Asegúrate de ejecutar primero:")
        logger.info(f"  python -m src.data.run_load_data")
        return 1
        
    except ValueError as e:
        logger.error(f"\n❌ ERROR: Valor inválido")
        logger.error(f"{str(e)}")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ ERROR INESPERADO:  {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())