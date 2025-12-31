"""
Esquemas Pydantic para validación de datos
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class HousingFeatures(BaseModel):
    """
    Features de entrada para predicción (ORIGINALES - 13 features)
    
    La API aplicará preprocessing automáticamente
    """
    
    # ======================================================================
    # SOLO FEATURES ORIGINALES (las que el usuario debe proporcionar)
    # ======================================================================
    
    CRIM: float = Field(..., description="Per capita crime rate", ge=0)
    ZN: float = Field(..., description="Proportion of residential land zoned for lots over 25,000 sq.ft.", ge=0, le=100)
    INDUS: float = Field(..., description="Proportion of non-retail business acres", ge=0, le=100)
    CHAS: int = Field(..., description="Charles River dummy variable (1 if bounds river; 0 otherwise)", ge=0, le=1)
    NOX: float = Field(..., description="Nitric oxides concentration (parts per 10 million)", ge=0, le=1)
    RM: float = Field(..., description="Average number of rooms per dwelling", ge=0, le=20)
    AGE: float = Field(..., description="Proportion of owner-occupied units built prior to 1940", ge=0, le=100)
    DIS: float = Field(..., description="Weighted distances to employment centres", ge=0)
    RAD: int = Field(..., description="Index of accessibility to radial highways", ge=1, le=24)
    TAX: float = Field(..., description="Property-tax rate per $10,000", ge=0)
    PTRATIO: float = Field(..., description="Pupil-teacher ratio", ge=0, le=50)
    B: float = Field(..., description="1000(Bk - 0.63)^2 where Bk is proportion of Black residents", ge=0)
    LSTAT: float = Field(..., description="% lower status of the population", ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "CRIM": 0.00632,
                "ZN": 18.0,
                "INDUS": 2.31,
                "CHAS": 0,
                "NOX": 0.538,
                "RM": 6.575,
                "AGE": 65.2,
                "DIS": 4.0900,
                "RAD":  1,
                "TAX": 296.0,
                "PTRATIO": 15.3,
                "B":  396.90,
                "LSTAT": 4.98
            }
        }
    
    def to_dataframe(self):
        """Convierte a DataFrame (solo features originales)"""
        import pandas as pd
        
        data = {
            'CRIM': [self.CRIM],
            'ZN': [self.ZN],
            'INDUS': [self.INDUS],
            'CHAS': [self.CHAS],
            'NOX': [self.NOX],
            'RM':  [self.RM],
            'AGE': [self.AGE],
            'DIS':  [self.DIS],
            'RAD': [self.RAD],
            'TAX':  [self.TAX],
            'PTRATIO': [self.PTRATIO],
            'B': [self.B],
            'LSTAT': [self.LSTAT]
        }
        
        return pd.DataFrame(data)


class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    predicted_price: float = Field(..., description="Precio predicho en miles de dólares")
    model_name: str = Field(..., description="Nombre del modelo usado")
    model_version: str = Field(..., description="Versión del modelo")
    
    class Config: 
        json_schema_extra = {
            "example": {
                "predicted_price": 24.5,
                "model_name": "xgboost",
                "model_version": "1.0.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request para predicción en batch"""
    instances: List[HousingFeatures] = Field(..., description="Lista de instancias")
    
    class Config:
        json_schema_extra = {
            "example":  {
                "instances": [
                    {
                        "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0,
                        "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.0900,
                        "RAD": 1, "TAX":  296.0, "PTRATIO": 15.3, "B": 396.90, "LSTAT": 4.98
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en batch"""
    predictions: List[float] = Field(..., description="Lista de precios predichos")
    count: int = Field(..., description="Número de predicciones")
    model_name: str = Field(..., description="Nombre del modelo usado")


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    preprocessors_loaded: bool = Field(..., description="Si los preprocessors están cargados")
    model_name: str = Field(..., description="Nombre del modelo")
    version: str = Field(..., description="Versión de la API")