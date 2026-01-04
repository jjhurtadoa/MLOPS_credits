"""
Módulo para carga y división de datos
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.utils.logger import get_logger
logger = get_logger(__name__)


def load_raw_data(data_path: str = 'data/raw/boston_housing.csv') -> pd.DataFrame:
    """Carga datos crudos desde CSV"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Archivo no encontrado: {data_path}")
    
    logger.info(f"📂 Cargando datos desde:  {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    logger.info(f"✓ Columnas: {list(df.columns)}")
    
    if df.empty:
        raise ValueError("❌ El dataset está vacío")
    
    logger.info(f"✓ Valores faltantes totales: {df.isnull().sum().sum()}")
    
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str = 'MEDV',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide datos en train y test"""
    logger.info("="*60)
    logger.info("DIVISIÓN TRAIN/TEST")
    logger.info("="*60)
    logger.info(f"Parámetros:")
    logger.info(f"  - Target:  {target_column}")
    logger.info(f"  - Test size: {test_size}")
    logger.info(f"  - Random state: {random_state}")
    logger.info(f"  - Stratify: {stratify}")
    
    if target_column not in df.columns:
        raise ValueError(f"❌ Columna target '{target_column}' no encontrada")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"✓ Train set: {X_train.shape[0]} filas ({X_train.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"✓ Test set:    {X_test.shape[0]} filas ({X_test.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"✓ Features:  {X_train.shape[1]}")
    logger.info(f"\nDistribución del target:")
    logger.info(f"  Train - Media: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    logger.info(f"  Test  - Media: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    return X_train, X_test, y_train, y_test


def save_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = 'data/splits',
    target_column: str = 'MEDV'
) -> None:
    """Guarda datos divididos en CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("GUARDANDO DATOS")
    logger.info("="*60)
    
    train_df = X_train.copy()
    train_df[target_column] = y_train.values
    
    test_df = X_test.copy()
    test_df[target_column] = y_test.values
    
    train_path = output_path / 'train.csv'
    test_path = output_path / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Corrección del error de formato
    train_size_kb = train_path.stat().st_size / 1024
    test_size_kb = test_path.stat().st_size / 1024
    
    logger.info(f"✓ Guardado:  {str(train_path)} ({train_size_kb:.1f} KB)")
    logger.info(f"✓ Guardado: {str(test_path)} ({test_size_kb:.1f} KB)")
    logger.info(f"✓ Columnas guardadas: {list(train_df.columns)}")


def load_split_data(
    data_dir: str = 'data/splits',
    target_column: str = 'MEDV'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga datos ya divididos"""
    data_path = Path(data_dir)
    
    train_path = data_path / 'train.csv'
    test_path = data_path / 'test.csv'
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"❌ Archivos no encontrados en {data_path}\n"
            f"   Ejecuta primero: python -m src.data.run_load_data"
        )
    
    logger.info(f"📂 Cargando datos procesados desde: {data_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    if target_column not in train_df.columns:
        raise ValueError(f"❌ Columna target '{target_column}' no encontrada")
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    logger.info(f"✓ Train:  {X_train.shape}")
    logger.info(f"✓ Test:  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test