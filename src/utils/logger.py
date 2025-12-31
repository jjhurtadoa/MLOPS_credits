"""
Configuración centralizada de logging para el proyecto
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Variable global para trackear si ya se configuró
_logging_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    force:  bool = False
) -> None:
    """
    Configura el sistema de logging globalmente
    
    Args: 
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Ruta opcional para guardar logs en archivo
        log_format:  Formato personalizado de logs
        force: Si True, reconfigura aunque ya esté configurado
    """
    global _logging_configured
    
    if _logging_configured and not force:
        logger = logging.getLogger(__name__)
        logger.debug("Logging ya configurado.  Use force=True para reconfigurar.")
        return
    
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Silenciar loggers ruidosos
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger para un módulo específico
    
    Args:
        name: Nombre del logger (usar __name__ del módulo)
    
    Returns:
        logging.Logger: Logger configurado
    """
    global _logging_configured
    
    if not _logging_configured: 
        setup_logging(level=logging.INFO)
    
    return logging. getLogger(name)


def reset_logging() -> None:
    """
    Resetea la configuración de logging
    
    Útil para testing cuando necesitas reconfigurar logging
    entre tests.
    
    Example:
        >>> # En pytest fixture
        >>> reset_logging()
        >>> setup_logging(level=logging.DEBUG)
    """
    global _logging_configured
    _logging_configured = False
    
    # Limpiar todos los handlers del root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Resetear nivel del root logger
    root_logger.setLevel(logging.WARNING)
    
    # Limpiar handlers de todos los loggers creados
    loggers_to_clear = [
        logging.getLogger(name) 
        for name in logging.root.manager.loggerDict
    ]
    for logger in loggers_to_clear:
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging. NOTSET)