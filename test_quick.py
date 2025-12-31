"""
Test rápido para desarrollo (sin pytest)
"""

# ✅ Configurar logging al inicio
from src.utils.logger import setup_logging, get_logger
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

from src.data import load_raw_data, split_data, save_data, load_processed_data
import pandas as pd

print("="*60)
print("🧪 TEST RÁPIDO:  src/data")
print("="*60)

try:
    # Test 1
    print("\n1️⃣ Cargando datos raw...")
    df = load_raw_data('data/raw/HousingData.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    print(f"   ✅ Cargado: {df.shape}")

    # Test 2
    print("\n2️⃣ Dividiendo datos...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(df)
    print(f"   ✅ Train: {len(X_train)}, Test: {len(X_test)}")

    # Test 3
    print("\n3️⃣ Guardando datos...")
    save_data(X_train, X_test, y_train, y_test)
    print("   ✅ Guardado exitoso")

    # Test 4
    print("\n4️⃣ Cargando datos procesados...")
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = load_processed_data()
    assert X_train. shape == X_train_loaded. shape
    print(f"   ✅ Cargado: Train {X_train_loaded.shape}, Test {X_test_loaded.shape}")

    print("\n" + "="*60)
    print("✅ TODOS LOS TESTS PASARON")
    print("="*60)

except Exception as e:
    logger.error(f"\n❌ ERROR:  {e}", exc_info=True)
    exit(1)