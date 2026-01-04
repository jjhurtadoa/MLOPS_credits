"""
Script ejecutable para entrenamiento de modelos

Flujo:  
    1. Cargar configuración
    2. Cargar datos procesados
    3. Entrenar modelos baseline (RandomForest, XGBoost, LightGBM, etc.)
    4. Comparar resultados
    5. Seleccionar mejor modelo
    6. (Opcional) Optimizar hiperparámetros del ganador con Optuna
    7. Guardar artifacts

Uso:
    # Baseline (rápido - ~5 min)
    python -m src.train.run_train
    
    # Baseline + Optimización (lento - ~60 min)
    python -m src.train.run_train --optimize --n-trials 100
    
    # Solo algunos modelos
    python -m src.train.run_train --models random_forest xgboost
    
    # Config personalizada
    python -m src. train.run_train --config custom_config.yaml
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
        '--optimize',
        action='store_true',
        help='Optimizar hiperparámetros del mejor modelo con Optuna'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Número de trials para optimización Optuna (default: 100)'
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
    if args.optimize:
        logger.info(f"⚡ Optimización habilitada (n_trials={args.n_trials})")
    
    try:
        # ================================================================
        # 1. CARGAR CONFIGURACIÓN
        # ================================================================
        
        config = load_config(args.config)
        
        # Filtrar modelos si se especificaron
        if args.models:
            logger.info(f"🎯 Entrenando solo:  {args.models}")
            models_to_train = {k: v for k, v in config['models'].items() if k in args.models}
            config['models'] = models_to_train
        
        # ================================================================
        # 2. INICIALIZAR TRAINER
        # ================================================================
        
        trainer = ModelTrainer(config)
        
        # ================================================================
        # 3. CARGAR DATOS
        # ================================================================
        
        X_train, X_test, y_train, y_test = trainer.load_data(data_dir=args.data_dir)
        
        # ================================================================
        # 4. ENTRENAR MODELOS BASELINE
        # ================================================================
        
        logger.info("\n" + "="*60)
        logger.info("📊 ENTRENAMIENTO BASELINE (parámetros por defecto)")
        logger.info("="*60)
        
        trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # ================================================================
        # 5. COMPARAR MODELOS
        # ================================================================
        
        comparison_df = trainer.compare_models()
        
        # Guardar comparación
        output_dir = Path(config['training']['model_dir'])
        comparison_path = output_dir / 'model_comparison_baseline.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\n💾 Comparación baseline guardada en:  {comparison_path}")
        
        # ================================================================
        # 6. MEJOR MODELO BASELINE
        # ================================================================
        
        best_name, best_model, best_metadata = trainer.get_best_model(metric='test_rmse')
        
        logger.info(f"\n🏆 Mejor modelo (baseline): {best_name}")
        logger.info(f"  - Test RMSE: {best_metadata['test_metrics']['rmse']:.4f}")
        logger.info(f"  - Test R²: {best_metadata['test_metrics']['r2']:.4f}")
        
        # Guardar modelo baseline
        baseline_model_path = output_dir / 'best_model_baseline.pkl'
        baseline_metadata_path = output_dir / 'best_model_baseline_metadata.yaml'
        
        import joblib
        joblib.dump(best_model, baseline_model_path)
        
        with open(baseline_metadata_path, 'w') as f:
            yaml.dump(best_metadata, f, default_flow_style=False)
        
        logger.info(f"\n💾 Modelo baseline guardado:")
        logger.info(f"  - Modelo: {baseline_model_path}")
        logger.info(f"  - Metadata: {baseline_metadata_path}")
        
        # ================================================================
        # 7. OPTIMIZACIÓN (OPCIONAL)
        # ================================================================
        
        final_model = best_model
        final_model_name = best_name
        final_metadata = best_metadata
        
        if args.optimize:
            logger.info("\n" + "="*60)
            logger.info("🔍 OPTIMIZACIÓN DE HIPERPARÁMETROS")
            logger.info("="*60)
            logger.info(f"Modelo a optimizar: {best_name}")
            logger.info(f"Trials: {args.n_trials}")
            
            try:
                from src.train.optimize_model import ModelOptimizer
                
                # Crear optimizador
                optimizer = ModelOptimizer(best_name, config)
                
                # Optimizar
                best_params = optimizer.optimize(X_train, y_train, n_trials=args.n_trials)
                
                # Entrenar modelo final con mejores parámetros
                optimized_model, optimized_metrics = optimizer.train_best_model(
                    X_train, y_train, X_test, y_test
                )
                
                # Actualizar modelo final
                final_model = optimized_model
                final_model_name = f"{best_name}_optimized"
                
                # Crear metadata del modelo optimizado
                final_metadata = {
                    'model_name': final_model_name,
                    'base_model': best_name,
                    'optimization':  'optuna',
                    'n_trials': args.n_trials,
                    'best_params': best_params,
                    'baseline_rmse': best_metadata['test_metrics']['rmse'],
                    'optimized_rmse': optimized_metrics['test_rmse'],
                    'improvement': best_metadata['test_metrics']['rmse'] - optimized_metrics['test_rmse'],
                    'test_metrics': {
                        'rmse': optimized_metrics['test_rmse'],
                        'mae': optimized_metrics['test_mae'],
                        'r2':  optimized_metrics['test_r2']
                    },
                    'train_metrics': {
                        'rmse': optimized_metrics['train_rmse'],
                        'mae': optimized_metrics['train_mae'],
                        'r2': optimized_metrics['train_r2']
                    }
                }
                
                logger.info(f"\n✅ Optimización completada")
                logger.info(f"  - Baseline RMSE: {best_metadata['test_metrics']['rmse']:.4f}")
                logger. info(f"  - Optimizado RMSE: {optimized_metrics['test_rmse']:. 4f}")
                logger.info(f"  - Mejora: {final_metadata['improvement']:.4f} ({final_metadata['improvement']/best_metadata['test_metrics']['rmse']*100:.1f}%)")
                
                # Guardar modelo optimizado separado
                optimized_model_path = output_dir / f'{best_name}_optimized.pkl'
                optimized_metadata_path = output_dir / f'{best_name}_optimized_metadata.yaml'
                
                joblib.dump(optimized_model, optimized_model_path)
                
                with open(optimized_metadata_path, 'w') as f:
                    yaml.dump(final_metadata, f, default_flow_style=False)
                
                logger.info(f"\n💾 Modelo optimizado guardado:")
                logger.info(f"  - Modelo: {optimized_model_path}")
                logger.info(f"  - Metadata: {optimized_metadata_path}")
                
            except ImportError:
                logger.warning("\n⚠️  Módulo 'optuna' no encontrado")
                logger.warning("  Instala con: pip install optuna")
                logger.warning("  Usando modelo baseline como final")
            
            except Exception as e: 
                logger.error(f"\n❌ Error en optimización: {str(e)}", exc_info=True)
                logger.warning("  Usando modelo baseline como final")
        
        # ================================================================
        # 8. GUARDAR MODELO FINAL (BEST_MODEL)
        # ================================================================
        
        best_model_path = output_dir / 'best_model. pkl'
        best_metadata_path = output_dir / 'best_model_metadata.yaml'
        
        joblib.dump(final_model, best_model_path)
        
        with open(best_metadata_path, 'w') as f:
            yaml.dump(final_metadata, f, default_flow_style=False)
        
        logger.info(f"\n💾 Modelo final guardado como 'best_model':")
        logger.info(f"  - Modelo: {best_model_path}")
        logger.info(f"  - Metadata: {best_metadata_path}")
        logger.info(f"  - Tipo: {'Optimizado' if args.optimize and 'optimized' in final_model_name else 'Baseline'}")
        
        # ================================================================
        # 9. RESUMEN FINAL
        # ================================================================
        
        logger. info("\n" + "="*60)
        logger.info("✅ TRAINING PIPELINE - COMPLETADO")
        logger.info("="*60)
        
        logger.info(f"\n📊 Resumen:")
        logger.info(f"  - Modelos baseline entrenados: {len(trainer. models_)}")
        logger.info(f"  - Mejor modelo baseline: {best_name}")
        
        if args.optimize and 'optimized' in final_model_name:
            logger.info(f"  - Modelo final: {final_model_name}")
            logger.info(f"  - RMSE baseline: {best_metadata['test_metrics']['rmse']:.4f}")
            logger.info(f"  - RMSE optimizado: {final_metadata['test_metrics']['rmse']:.4f}")
            logger. info(f"  - Mejora: {final_metadata['improvement']:.4f} ({final_metadata['improvement']/best_metadata['test_metrics']['rmse']*100:.1f}%)")
        else:
            logger.info(f"  - Test RMSE: {final_metadata['test_metrics']['rmse']:.4f}")
            logger.info(f"  - Test R²: {final_metadata['test_metrics']['r2']:. 4f}")
        
        logger.info(f"\n📁 Artifacts generados:")
        logger.info(f"  - Modelos baseline: {output_dir}/*_model.pkl")
        logger.info(f"  - Metadata: {output_dir}/*_metadata.yaml")
        logger.info(f"  - Comparación: {comparison_path}")
        logger.info(f"  - Best model: {best_model_path}")
        if args.optimize:
            logger.info(f"  - Modelo optimizado: {output_dir}/{best_name}_optimized.pkl")
        logger.info(f"  - MLflow: {config['training']['mlflow']['tracking_uri']}")
        
        logger.info(f"\n🔜 Siguiente paso:")
        logger.info(f"  python -m src. evaluate.run_evaluate --model {best_model_path}")
        logger.info(f"  mlflow ui --backend-store-uri {config['training']['mlflow']['tracking_uri']}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR:  Archivo no encontrado")
        logger.error(f"{str(e)}")
        logger.info(f"\n💡 Asegúrate de ejecutar primero:")
        logger.info(f"  python -m src. preprocess.run_preprocess")
        return 1
        
    except Exception as e: 
        logger.error(f"\n❌ ERROR INESPERADO:  {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())