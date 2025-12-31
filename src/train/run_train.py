"""
Script ejecutable para entrenamiento de modelos

Flujo: 
    1.Cargar configuración
    2.Cargar datos procesados
    3.Entrenar modelos (RandomForest, XGBoost, LightGBM)
    4.Comparar resultados
    5.Seleccionar mejor modelo
    6.Guardar artifacts

Uso:
    python -m src.train.run_train
    python -m src.train.run_train --config custom_config.yaml
    python -m src.train.run_train --models random_forest xgboost
"""

import argparse
import yaml
import sys
from pathlib import Path

from src.utils.logger import setup_logging, get_logger
from src.train.train_model import ModelTrainer

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    Carga configuración desde YAML
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns: 
        dict: Configuración cargada
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config no encontrado: {config_path}")
    
    logger.info(f"📄 Cargando config desde: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✓ Config cargado")
    
    return config


def main():
    """Pipeline principal de entrenamiento"""
    parser = argparse.ArgumentParser(
        description='Pipeline de entrenamiento de modelos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/model_config.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/interim',
        help='Directorio con datos procesados'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Modelos específicos a entrenar (ej: random_forest xgboost)'
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
        default='artifacts/logs/training.log',
        help='Archivo de log'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    import logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(level=log_level, log_file=args.log_file)
    
    logger.info("\n" + "="*60)
    logger.info("🚀 TRAINING PIPELINE - INICIO")
    logger.info("="*60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    
    try:
        # ================================================================
        # 1.CARGAR CONFIGURACIÓN
        # ================================================================
        
        config = load_config(args.config)
        
        # Filtrar modelos si se especificaron
        if args.models:
            logger.info(f"🎯 Entrenando solo:  {args.models}")
            models_to_train = {k: v for k, v in config['models'].items() if k in args.models}
            config['models'] = models_to_train
        
        # ================================================================
        # 2.INICIALIZAR TRAINER
        # ================================================================
        
        trainer = ModelTrainer(config)
        
        # ================================================================
        # 3.CARGAR DATOS
        # ================================================================
        
        X_train, X_test, y_train, y_test = trainer.load_data(data_dir=args.data_dir)
        
        # ================================================================
        # 4.ENTRENAR MODELOS
        # ================================================================
        
        trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # ================================================================
        # 5.COMPARAR MODELOS
        # ================================================================
        
        comparison_df = trainer.compare_models()
        
        # Guardar comparación
        output_dir = Path(config['training']['model_dir'])
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\n💾 Comparación guardada en:  {comparison_path}")
        
        # ================================================================
        # 6.MEJOR MODELO
        # ================================================================
        
        best_name, best_model, best_metadata = trainer.get_best_model(metric='test_rmse')
        
        # Guardar como "best_model"
        best_model_path = output_dir / 'best_model.pkl'
        import joblib
        joblib.dump(best_model, best_model_path)
        
        best_metadata_path = output_dir / 'best_model_metadata.yaml'
        with open(best_metadata_path, 'w') as f:
            yaml.dump(best_metadata, f, default_flow_style=False)
        
        logger.info(f"\n💾 Mejor modelo guardado:")
        logger.info(f"  - Modelo: {best_model_path}")
        logger.info(f"  - Metadata:  {best_metadata_path}")
        
        # ================================================================
        # 7.RESUMEN FINAL
        # ================================================================
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING PIPELINE - COMPLETADO")
        logger.info("="*60)
        
        logger.info(f"\n📊 Resumen:")
        logger.info(f"  - Modelos entrenados: {len(trainer.models_)}")
        logger.info(f"  - Mejor modelo: {best_name}")
        logger.info(f"  - Test RMSE: {best_metadata['test_metrics']['rmse']:.4f}")
        logger.info(f"  - Test R²: {best_metadata['test_metrics']['r2']:.4f}")
        
        logger.info(f"\n📁 Artifacts generados:")
        logger.info(f"  - Modelos:  {output_dir}/*_model.pkl")
        logger.info(f"  - Metadata: {output_dir}/*_metadata.yaml")
        logger.info(f"  - Comparación: {comparison_path}")
        logger.info(f"  - MLflow:  {config['training']['mlflow']['tracking_uri']}")
        
        logger.info(f"\n🔜 Siguiente paso:")
        logger.info(f"  python -m src.evaluate.run_evaluate")
        logger.info(f"  mlflow ui --backend-store-uri {config['training']['mlflow']['tracking_uri']}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR:  Archivo no encontrado")
        logger.error(f"{str(e)}")
        logger.info(f"\n💡 Asegúrate de ejecutar primero:")
        logger.info(f"  python -m src.preprocess.run_preprocess")
        return 1
        
    except Exception as e: 
        logger.error(f"\n❌ ERROR INESPERADO:  {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())