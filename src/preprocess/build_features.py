"""
Módulo de Feature Engineering

Crea features derivadas y transformaciones adicionales
para mejorar el poder predictivo del modelo.

Estrategias implementadas:
- Features de interacción (productos, ratios)
- Features polinomiales
- Binning de variables continuas
- Features de dominio específico (real estate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# INTERACTION FEATURES
# ============================================================================

class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Crea features de interacción entre variables existentes
    
    Tipos de interacción:
    - Producto: feature1 * feature2
    - Ratio: feature1 / feature2
    - Diferencia: feature1 - feature2
    - Suma: feature1 + feature2
    
    Attributes:
        interactions (list): Lista de interacciones a crear
        feature_names_out_ (list): Nombres de features generadas

]
    """
    
    def __init__(self, interactions:  Optional[List[Dict]] = None):
        """
        Args:
            interactions:  Lista de diccionarios con especificaciones de interacción
                Formato: {'type': 'product'|'ratio'|'diff'|'sum', 
                         'features': [col1, col2], 
                         'name': 'nombre_nuevo'}
        """
        self.interactions = interactions or []
        self.feature_names_in_ = None
        self.feature_names_out_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Valida que las features de interacción existen
        
        Args:
            X:  DataFrame de features
            y: Target (no usado)
            
        Returns:
            self:  Objeto fitted
        """
        logger.info("="*60)
        logger.info("FITTING INTERACTION FEATURE CREATOR")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Validar interacciones
        valid_interactions = []
        for interaction in self.interactions:
            required_features = interaction.get('features', [])
            
            # Verificar que features existen
            missing = [f for f in required_features if f not in X.columns]
            if missing:
                logger.warning(f"⚠️  Interacción '{interaction.get('name')}' omitida:  "
                             f"features faltantes {missing}")
                continue
            
            valid_interactions.append(interaction)
            self.feature_names_out_.append(interaction.get('name'))
        
        self.interactions = valid_interactions
        
        if self.interactions:
            logger.info(f"✓ {len(self.interactions)} interacciones configuradas")
            for interaction in self.interactions:
                logger.debug(f"  - {interaction['name']}: "
                           f"{interaction['type']}({interaction['features']})")
        else:
            logger.info("✓ No se crearán features de interacción")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de interacción
        
        Args:
            X: DataFrame original
            
        Returns:
            DataFrame con features originales + interacciones
        """
        X_transformed = X.copy()
        
        for interaction in self.interactions:
            interaction_type = interaction['type']
            features = interaction['features']
            name = interaction['name']
            
            try:
                if interaction_type == 'product': 
                    X_transformed[name] = X[features[0]] * X[features[1]]
                    
                elif interaction_type == 'ratio':
                    # Evitar división por cero
                    denominator = X[features[1]].replace(0, np.nan)
                    X_transformed[name] = X[features[0]] / denominator
                    
                elif interaction_type == 'diff':
                    X_transformed[name] = X[features[0]] - X[features[1]]
                    
                elif interaction_type == 'sum':
                    X_transformed[name] = X[features[0]] + X[features[1]]
                    
                else: 
                    logger.warning(f"⚠️  Tipo de interacción '{interaction_type}' no reconocido")
                    continue
                
                logger.debug(f"✓ Creada feature: {name}")
                
            except Exception as e: 
                logger.error(f"❌ Error creando {name}: {e}")
        
        if self.interactions:
            logger.info(f"✓ Creadas {len(self.interactions)} features de interacción")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Retorna nombres de todas las features (originales + nuevas)"""
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + self.feature_names_out_


# ============================================================================
# DOMAIN-SPECIFIC FEATURES (Real Estate)
# ============================================================================

class RealEstateDomainFeatures(BaseEstimator, TransformerMixin):
    """
    Crea features específicas del dominio de bienes raíces
    
    Features creadas:
    - rooms_per_age: Calidad de mantenimiento (RM / AGE)
    - tax_per_room: Impuesto por habitación (TAX / RM)
    - accessibility_score: Score de accesibilidad (DIS * RAD)
    - socioeconomic_index: Índice socioeconómico (LSTAT * PTRATIO)
    - property_quality: Calidad de propiedad (RM * (1 - LSTAT/100))
    
    Estas features son domain knowledge basado en: 
    - Teoría de valuación de propiedades
    - Factores que afectan precios de vivienda
    - Combinaciones que capturan conceptos de alto nivel
    """
    
    def __init__(self, create_all:  bool = True):
        """
        Args:
            create_all: Si False, permite seleccionar features específicas
        """
        self.create_all = create_all
        self.feature_names_in_ = None
        self.created_features_ = []
        
    def fit(self, X:  pd.DataFrame, y=None):
        """Valida que features necesarias existen"""
        logger.info("="*60)
        logger.info("FITTING REAL ESTATE DOMAIN FEATURES")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Features requeridas para cada nueva feature
        requirements = {
            'rooms_per_age': ['RM', 'AGE'],
            'tax_per_room': ['TAX', 'RM'],
            'accessibility_score': ['DIS', 'RAD'],
            'socioeconomic_index':  ['LSTAT', 'PTRATIO'],
            'property_quality': ['RM', 'LSTAT'],
            'crime_per_capita_weighted': ['CRIM', 'INDUS'],
            'pollution_residential':  ['NOX', 'INDUS']
        }
        
        # Determinar qué features se pueden crear
        self.created_features_ = []
        for feature_name, required_cols in requirements.items():
            if all(col in X.columns for col in required_cols):
                self.created_features_.append(feature_name)
            else:
                missing = [c for c in required_cols if c not in X.columns]
                logger.warning(f"⚠️  No se creará '{feature_name}':  faltan {missing}")
        
        logger.info(f"✓ Se crearán {len(self.created_features_)} domain features")
        logger.debug(f"  Features:  {self.created_features_}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Crea features de dominio"""
        X_transformed = X.copy()
        
        # 1.Rooms per Age (calidad de mantenimiento)
        if 'rooms_per_age' in self.created_features_:
            # Más habitaciones en edificio viejo = mejor mantenimiento
            X_transformed['rooms_per_age'] = X['RM'] / (X['AGE'] + 1)  # +1 para evitar div/0
            logger.debug("✓ Creada:  rooms_per_age (RM/AGE)")
        
        # 2.Tax per Room
        if 'tax_per_room' in self.created_features_:
            X_transformed['tax_per_room'] = X['TAX'] / X['RM']
            logger.debug("✓ Creada: tax_per_room (TAX/RM)")
        
        # 3.Accessibility Score
        if 'accessibility_score' in self.created_features_: 
            # Menor distancia + más acceso a highways = mejor
            X_transformed['accessibility_score'] = X['RAD'] / (X['DIS'] + 0.1)
            logger.debug("✓ Creada:  accessibility_score (RAD/DIS)")
        
        # 4.Socioeconomic Index
        if 'socioeconomic_index' in self.created_features_: 
            # Combina % población de bajo estatus y ratio estudiante-profesor
            X_transformed['socioeconomic_index'] = X['LSTAT'] * X['PTRATIO']
            logger.debug("✓ Creada: socioeconomic_index (LSTAT*PTRATIO)")
        
        # 5.Property Quality
        if 'property_quality' in self.created_features_: 
            # Más habitaciones + menos población baja = mejor calidad
            X_transformed['property_quality'] = X['RM'] * (1 - X['LSTAT']/100)
            logger.debug("✓ Creada:  property_quality (RM*(1-LSTAT/100))")
        
        # 6.Crime weighted by industry
        if 'crime_per_capita_weighted' in self.created_features_:
            X_transformed['crime_per_capita_weighted'] = X['CRIM'] * X['INDUS']
            logger.debug("✓ Creada: crime_per_capita_weighted (CRIM*INDUS)")
        
        # 7.Pollution in residential areas
        if 'pollution_residential' in self.created_features_:
            # NOx en áreas no industriales es peor
            X_transformed['pollution_residential'] = X['NOX'] * (1 - X['INDUS']/100)
            logger.debug("✓ Creada:  pollution_residential (NOX*(1-INDUS/100))")
        
        logger.info(f"✓ Creadas {len(self.created_features_)} domain features")
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Retorna features originales + domain features"""
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + self.created_features_


# ============================================================================
# POLYNOMIAL FEATURES
# ============================================================================

class PolynomialFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Crea features polinomiales (cuadrados, cubos, interacciones)
    
    Wrapper de sklearn.preprocessing.PolynomialFeatures con logging
    
    Attributes:
        degree (int): Grado del polinomio
        interaction_only (bool): Solo interacciones, no potencias
        include_bias (bool): Incluir término constante
    """
    
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        features_to_transform: Optional[List[str]] = None
    ):
        """
        Args:
            degree:  Grado del polinomio (2 = cuadrados, 3 = cubos)
            interaction_only: Si True, solo crea interacciones (no x^2)
            include_bias:  Incluir columna de 1s
            features_to_transform:  Subset de features (None = todas)
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.features_to_transform = features_to_transform
        self.poly_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.other_features_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Entrena PolynomialFeatures"""
        logger.info("="*60)
        logger.info(f"FITTING POLYNOMIAL FEATURES (degree={self.degree})")
        logger.info("="*60)
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Determinar qué features transformar
        if self.features_to_transform: 
            transform_cols = [c for c in self.features_to_transform if c in X.columns]
            self.other_features_ = [c for c in X.columns if c not in transform_cols]
        else:
            transform_cols = X.columns.tolist()
            self.other_features_ = []
        
        if not transform_cols:
            logger.warning("⚠️  No hay features para transformar")
            return self
        
        # Crear y entrenar PolynomialFeatures
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        self.poly_.fit(X[transform_cols])
        
        # Obtener nombres de features generadas
        poly_feature_names = self.poly_.get_feature_names_out(transform_cols)
        self.feature_names_out_ = list(self.other_features_) + list(poly_feature_names)
        
        n_new_features = len(poly_feature_names) - len(transform_cols)
        logger.info(f"✓ Polynomial features creadas:")
        logger.info(f"  - Input features: {len(transform_cols)}")
        logger.info(f"  - Output features: {len(poly_feature_names)}")
        logger.info(f"  - New features: {n_new_features}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformación polinomial"""
        if self.poly_ is None:
            logger.warning("⚠️  PolynomialFeatures no fitted, retornando datos sin cambios")
            return X.copy()
        
        # Transformar subset
        if self.features_to_transform:
            transform_cols = [c for c in self.features_to_transform if c in X.columns]
        else:
            transform_cols = X.columns.tolist()
        
        X_poly = self.poly_.transform(X[transform_cols])
        
        # Convertir a DataFrame
        poly_feature_names = self.poly_.get_feature_names_out(transform_cols)
        X_poly_df = pd.DataFrame(
            X_poly,
            columns=poly_feature_names,
            index=X.index
        )
        
        # Agregar features no transformadas
        if self.other_features_: 
            X_poly_df = pd.concat([X[self.other_features_], X_poly_df], axis=1)
        
        logger.info(f"✓ Transformación polinomial aplicada:  {X.shape} -> {X_poly_df.shape}")
        
        return X_poly_df
    
    def get_feature_names_out(self, input_features=None):
        """Retorna nombres de features polinomiales"""
        return self.feature_names_out_


# ============================================================================
# FEATURE ENGINEER (Pipeline Completo)
# ============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Pipeline completo de feature engineering
    
    Aplica en orden: 
    1.Features de interacción (si están configuradas)
    2.Features de dominio (real estate)
    3.Features polinomiales (opcional)
    
    Attributes:
        config (dict): Configuración de feature engineering
        interaction_creator_ (InteractionFeatureCreator): Fitted
        domain_features_ (RealEstateDomainFeatures): Fitted
        poly_features_ (PolynomialFeatureCreator): Fitted (opcional)
        
    Example:
        >>> config = {
        ...'interactions': [        ...{'type': 'product', 'features': ['RM', 'AGE'], 'name': 'RM_x_AGE'},
        ...{'type': 'ratio', 'features': ['LSTAT', 'RM'], 'name': 'LSTAT_per_RM'}
        ...],
        ...'domain_features': True,
        ...'polynomial':  {'degree': 2, 'features': ['RM', 'LSTAT']}
        ...}
    """
    
    def __init__(self, config:  Optional[Dict] = None):
        """
        Args:
            config:  Configuración de feature engineering
        """
        self.config = config or {}
        self.interaction_creator_ = None
        self.domain_features_ = None
        self.poly_features_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Entrena todos los creadores de features"""
        logger.info("\n" + "="*60)
        logger.info("🔧 FITTING FEATURE ENGINEER")
        logger.info("="*60)
        logger.info(f"Input shape: {X.shape}")
        
        self.feature_names_in_ = X.columns.tolist()
        X_temp = X.copy()
        
        # 1.Interaction Features
        if 'interactions' in self.config and self.config['interactions']:
            self.interaction_creator_ = InteractionFeatureCreator(
                self.config['interactions']
            )
            X_temp = self.interaction_creator_.fit_transform(X_temp)
        
        # 2.Domain Features
        if self.config.get('domain_features', True):
            self.domain_features_ = RealEstateDomainFeatures()
            X_temp = self.domain_features_.fit_transform(X_temp)
        
        # 3.Polynomial Features (opcional)
        if 'polynomial' in self.config:
            poly_config = self.config['polynomial']
            self.poly_features_ = PolynomialFeatureCreator(
                degree=poly_config.get('degree', 2),
                interaction_only=poly_config.get('interaction_only', False),
                features_to_transform=poly_config.get('features', None)
            )
            X_temp = self.poly_features_.fit_transform(X_temp)
        
        self.feature_names_out_ = X_temp.columns.tolist()
        
        logger.info("\n" + "="*60)
        logger.info("✅ FEATURE ENGINEERING FITTED")
        logger.info("="*60)
        logger.info(f"Output shape: {X_temp.shape}")
        logger.info(f"Features added: {X_temp.shape[1] - X.shape[1]}")
        logger.info(f"New features: {set(self.feature_names_out_) - set(self.feature_names_in_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica feature engineering"""
        logger.info(f"\n🔄 Aplicando feature engineering:  {X.shape}")
        
        X_transformed = X.copy()
        
        if self.interaction_creator_: 
            X_transformed = self.interaction_creator_.transform(X_transformed)
            
        if self.domain_features_:
            X_transformed = self.domain_features_.transform(X_transformed)
            
        if self.poly_features_: 
            X_transformed = self.poly_features_.transform(X_transformed)
        
        logger.info(f"✓ Feature engineering completado: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit y transform en un paso"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Retorna nombres de features finales"""
        return self.feature_names_out_
    
    def save(self, filepath: str) -> None:
        """Guarda feature engineer"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"💾 FeatureEngineer guardado en:  {filepath}")
    
    @staticmethod
    def load(filepath:  str) -> 'FeatureEngineer':
        """Carga feature engineer"""
        engineer = joblib.load(filepath)
        logger.info(f"📂 FeatureEngineer cargado desde: {filepath}")
        return engineer