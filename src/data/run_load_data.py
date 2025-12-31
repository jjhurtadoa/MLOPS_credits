"""
Script ejecutable para pipeline de carga y división de datos
"""

import argparse
from pathlib import Path

# ✅ ENTRYPOINT: Aquí SÍ configuramos logging
from src.utils.logger import setup_logging, get_logger
from . load_data import load_raw_data, split_data, save_data


def main():
    """Pipeline principal de carga y división de datos"""
    parser = argparse.ArgumentParser(
        description='Pipeline de carga y división de datos',
        formatter_class=argparse. ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default='data/raw/HousingData.csv')
    parser.add_argument('--output', type=str, default='data/processed')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--target', type=str, default='MEDV')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file', type=str, default=None,
                        help='Ruta para guardar logs (opcional)')
    
    args = parser.parse_args()
    
    # ✅ CONFIGURAR LOGGING UNA SOLA VEZ (en el entrypoint)
    import logging
    log_level = getattr(logging, args. log_level.upper())
    setup_logging(level=log_level, log_file=args.log_file)
    
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("🚀 PIPELINE DE DATOS - INICIO")
    logger.info("="*60)
    
    try:
        # 1. Cargar datos raw
        df = load_raw_data(args.input)
        
        # 2. Dividir train/test
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_column=args. target,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # 3. Guardar
        save_data(
            X_train, X_test, y_train, y_test,
            output_dir=args.output,
            target_column=args.target
        )
        
        logger.info("\n" + "="*60)
        logger.info("✅ PIPELINE DE DATOS - COMPLETADO")
        logger.info("="*60)
        logger.info(f"\nArchivos generados:")
        logger.info(f"  - {args.output}/train.csv")
        logger.info(f"  - {args.output}/test.csv")
        logger.info(f"\nSiguiente paso:")
        logger.info(f"  python -m src.preprocess.run_preprocess")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()