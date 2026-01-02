"""
FastAPI - Boston Housing Price Prediction API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import pandas as pd
import logging

from api.config import settings
from api.schemas import (
    HousingFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from api.utils import model_loader, preprocessor_loader

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN CONTEXT (reemplaza on_event)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager para startup/shutdown
    
    Se ejecuta: 
    - Al inicio:  carga modelo y preprocessors
    - Al final: cleanup (si fuera necesario)
    """
    # STARTUP
    try:
        logger.info("🚀 Iniciando API...")
        
        # Cargar preprocessors
        preprocessor_loader.load_preprocessors(
            "artifacts/preprocessors/data_preprocessor.pkl",
            "artifacts/preprocessors/feature_engineer.pkl"
        )
        
        # Cargar modelo
        model_loader.load_model("artifacts/models/best_model.pkl")
        
        logger.info("✓ API lista")
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise
    
    # Yield control to app
    yield
    
    # SHUTDOWN (si fuera necesario)
    logger.info("👋 Cerrando API...")


# Crear app con lifespan
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # ← Nuevo
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS (sin cambios)
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Información básica de la API"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "model":   model_loader.get_model_name(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if (model_loader.is_loaded and preprocessor_loader.is_loaded) else "unhealthy",
        model_loaded=model_loader.is_loaded,
        preprocessors_loaded=preprocessor_loader.is_loaded,  # ← Cambiar nombre aquí
        model_name=model_loader.get_model_name(),
        version=settings.app_version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: HousingFeatures):
    """Predice el precio de una vivienda"""
    try: 
        # 1.Convertir a DataFrame (features originales)
        X_raw = features.to_dataframe()
        
        logger.info(f"Features originales recibidas: {X_raw.columns.tolist()}")
        
        # 2.Aplicar preprocessing
        X_processed = preprocessor_loader.preprocess(X_raw)
        
        logger.info(f"Features procesadas: {X_processed.columns.tolist()}")
        
        # 3.Predecir
        model = model_loader.load_model(settings.model_path)
        prediction = model.predict(X_processed)[0]
        
        logger.info(f"Predicción exitosa: {prediction:.2f}")
        
        return PredictionResponse(
            predicted_price=float(prediction),
            model_name=model_loader.get_model_name(),
            model_version=settings.app_version
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en predicción: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predice precios para múltiples viviendas"""
    try:
        # Validar que hay instancias
        if not request.instances:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Lista de instancias no puede estar vacía"
            )
        
        # 1.Convertir instancias a DataFrame
        dfs = [instance.to_dataframe() for instance in request.instances]
        X_raw = pd.concat(dfs, ignore_index=True)
        
        # 2.Aplicar preprocessing
        X_processed = preprocessor_loader.preprocess(X_raw)
        
        # 3.Predecir
        model = model_loader.load_model(settings.model_path)
        predictions = model.predict(X_processed)
        
        logger.info(f"Batch prediction exitoso: {len(predictions)} predicciones")
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            count=len(predictions),
            model_name=model_loader.get_model_name()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en batch prediction: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handler para errores de validación"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail":  str(exc)}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handler genérico"""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )