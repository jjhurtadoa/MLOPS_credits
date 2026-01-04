"""
FastAPI - Boston Housing Price Prediction API
"""

from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException, status, Request  # ← AGREGAR Request
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

# ============================================================================
# NUEVAS IMPORTS PARA RATE LIMITING Y PROMETHEUS
# ============================================================================
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ============================================================================
# LOGGING
# ============================================================================

# Crear carpeta de logs
Path("artifacts/logs").mkdir(parents=True, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('artifacts/logs/api.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITER
# ============================================================================

limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# LIFESPAN CONTEXT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager para startup/shutdown"""
    # STARTUP
    try:
        logger.info("[OK] Iniciando API...")
        
        # Cargar preprocessors
        preprocessor_loader.load_preprocessors(
            "artifacts/preprocessors/data_preprocessor.pkl",
            "artifacts/preprocessors/feature_engineer.pkl"
        )
        
        # Cargar modelo
        model_loader.load_model("artifacts/models/best_model.pkl")
        
        logger.info("[OK] API lista")
    except Exception as e:
        logger.error(f"[ERROR] Error en startup: {e}")
        raise
    
    yield
    
    # SHUTDOWN
    logger.info("[INFO] Cerrando API...")


# Crear app con lifespan
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ============================================================================
# RATE LIMITER
# ============================================================================

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PROMETHEUS
# ============================================================================

Instrumentator().instrument(app).expose(app)
logger.info("[OK] Prometheus metrics habilitado en /metrics")

# ============================================================================
# ENDPOINTS
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
            "batch_predict": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if (model_loader.is_loaded and preprocessor_loader.is_loaded) else "unhealthy",
        model_loaded=model_loader.is_loaded,
        preprocessors_loaded=preprocessor_loader.is_loaded,
        model_name=model_loader.get_model_name(),
        version=settings.app_version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("10/minute")  # ← RATE LIMITING
async def predict(request: Request, features: HousingFeatures):  # ← request:  Request
    """
    Predice el precio de una vivienda
    
    **Rate limit:** 10 requests por minuto por IP
    """
    try:  
        # 1.Convertir a DataFrame (features originales)
        X_raw = features.to_dataframe()
        
        logger.info(f"Features originales recibidas: {X_raw.columns.tolist()}")
        
        # 2.Aplicar preprocessing
        X_processed = preprocessor_loader.preprocess(X_raw)
        
        logger.info(f"Features procesadas: {X_processed.columns.tolist()}")
        
        # 3.Predecir (IGUAL QUE TU CÓDIGO ORIGINAL)
        model = model_loader.load_model(settings.model_path)
        prediction = model.predict(X_processed)[0]
        
        logger.info(f"Prediccion exitosa: {prediction:.2f}")
        
        return PredictionResponse(
            predicted_price=float(prediction),  # ← predicted_price (como tu schema)
            model_name=model_loader.get_model_name(),
            model_version=settings.app_version
        )
        
    except Exception as e: 
        logger.error(f"Error en prediccion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en prediccion: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
@limiter.limit("5/minute")  # ← RATE LIMITING (más estricto)
async def predict_batch(request: Request, batch_request: BatchPredictionRequest):  # ← request: Request
    """
    Predice precios para múltiples viviendas
    
    **Rate limit:** 5 requests por minuto por IP
    """
    try:
        # Validar que hay instancias
        if not batch_request.instances:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Lista de instancias no puede estar vacia"
            )
        
        # 1.Convertir instancias a DataFrame
        dfs = [instance.to_dataframe() for instance in batch_request.instances]
        X_raw = pd.concat(dfs, ignore_index=True)
        
        # 2.Aplicar preprocessing
        X_processed = preprocessor_loader.preprocess(X_raw)
        
        # 3.Predecir (IGUAL QUE TU CÓDIGO ORIGINAL)
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
    """Handler para errores de validacion"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handler generico"""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )