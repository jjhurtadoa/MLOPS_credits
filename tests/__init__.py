"""
Test suite para el proyecto Boston Housing MLOps
"""

# Permite importar desde src/ en los tests
import sys
from pathlib import Path

# Añadir src/ al path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))