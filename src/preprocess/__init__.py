"""Paquete `src.preprocess` — transformaciones y pipeline de preprocesamiento.

Exporta las clases públicas principales (DataPreprocessor, handlers y feature-engineers).
Evitar ejecutar código costoso al importar este paquete.
"""

from .preprocessing import (
    DataPreprocessor,
    MissingValueHandler,
    OutlierHandler,
    CorrelationReducer,
    FeatureScaler,
)

from .build_features import (
    InteractionFeatureCreator,
    RealEstateDomainFeatures,
    PolynomialFeatureCreator,
    FeatureEngineer,
)

__all__ = [
    'DataPreprocessor',
    'MissingValueHandler',
    'OutlierHandler',
    'CorrelationReducer',
    'FeatureScaler',
    'InteractionFeatureCreator',
    'RealEstateDomainFeatures',
    'PolynomialFeatureCreator',
    'FeatureEngineer',
]
