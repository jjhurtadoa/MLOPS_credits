"""Paquete `src.preprocess` — transformaciones y pipeline de preprocesamiento.

Exporta las clases públicas principales (DataPreprocessor y handlers).
Evitar ejecutar código costoso al importar este paquete.
"""

from .preprocessing import (
    DataPreprocessor,
    MissingValueHandler,
    OutlierHandler,
    CorrelationReducer,
    FeatureScaler
)

__all__ = [
    'DataPreprocessor',
    'MissingValueHandler',
    'OutlierHandler',
    'CorrelationReducer',
    'FeatureScaler',
]
