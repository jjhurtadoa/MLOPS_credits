"""
Script ejecutable para evaluación de modelos

Usage:
    python -m src.evaluate.run_evaluate --model artifacts/models/best_model.pkl
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

from src.evaluate.evaluate_model import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Evaluación de modelos de ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Evaluar mejor modelo
  python -m src.evaluate.run_evaluate --model artifacts/models/best_model.pkl
  
  # Evaluar modelo específico
  python -m src.evaluate.run_evaluate --model artifacts/models/xgboost_model.pkl
  
  # Evaluar con datos de train y test
  python -m src.evaluate.run_evaluate --model artifacts/models/best_model.pkl --include-train
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo .pkl'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/interim',
        help='Directorio con datos procesados (default: data/interim)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/evaluation',
        help='Directorio de salida para reportes (default: artifacts/evaluation)'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='MEDV',
        help='Nombre de la columna target (default: MEDV)'
    )
    
    parser.add_argument(
        '--include-train',
        action='store_true',
        help='Incluir evaluación en datos de entrenamiento'
    )
    
    return parser.parse_args()


def main():
    """Función principal"""
    args = parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("🚀 EVALUATION PIPELINE - INICIO")
    logger.info("="*60)
    logger.info(f"Modelo: {args.model}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Validar que existe el modelo
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ Modelo no encontrado:  {model_path}")
        sys.exit(1)
    
    # Cargar datos de test
    data_dir = Path(args.data_dir)
    test_path = data_dir / 'test_processed.csv'
    
    if not test_path.exists():
        logger.error(f"❌ Datos de test no encontrados:  {test_path}")
        sys.exit(1)
    
    logger.info(f"\n📂 Cargando datos de test: {test_path}")
    test_df = pd.read_csv(test_path)
    
    X_test = test_df.drop(columns=[args.target_column])
    y_test = test_df[args.target_column]
    
    logger.info(f"✓ Test set cargado: {X_test.shape}")
    
    # Cargar datos de train (opcional)
    X_train, y_train = None, None
    if args.include_train:
        train_path = data_dir / 'train_processed.csv'
        
        if not train_path.exists():
            logger.warning(f"⚠️ Datos de train no encontrados:  {train_path}")
        else:
            logger.info(f"\n📂 Cargando datos de train: {train_path}")
            train_df = pd.read_csv(train_path)
            
            X_train = train_df.drop(columns=[args.target_column])
            y_train = train_df[args.target_column]
            
            logger.info(f"✓ Train set cargado: {X_train.shape}")
    
    # Inicializar evaluador
    evaluator = ModelEvaluator(args.model)
    
    # Evaluar
    evaluator.evaluate(X_test, y_test, X_train, y_train)
    
    # Generar reporte
    evaluator.generate_report(args.output_dir, X_test)
    
    logger.info("\n" + "="*60)
    logger.info("✅ EVALUATION PIPELINE - COMPLETADO")
    logger.info("="*60)
    logger.info(f"\n📂 Resultados en: {args.output_dir}")
    logger.info(f"  - Métricas: {args.output_dir}/{evaluator.model_name}_metrics.json")
    logger.info(f"  - Gráficos: {args.output_dir}/{evaluator.model_name}_*.png")


if __name__ == '__main__':
    main()