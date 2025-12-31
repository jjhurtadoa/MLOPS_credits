"""
Visualizaciones para evaluación de modelos

Funciones: 
- plot_residuals: Residual plot
- plot_predictions: Predicted vs Actual
- plot_feature_importance: Feature importance
- plot_error_distribution: Distribution of errors
- plot_model_comparison:  Comparación entre modelos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path:  Optional[Path] = None
) -> plt.Figure:
    """
    Gráfico de residuales
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        title: Título del gráfico
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{title} - Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of residuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{title} - Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado:  {save_path}")
    
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Gráfico de predicciones vs valores reales
    
    Args: 
        y_true: Valores reales
        y_pred:  Predicciones
        title:  Título
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {save_path}")
    
    return fig


def plot_feature_importance(
    model:  Any,
    feature_names: list,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Gráfico de importancia de features
    
    Args:
        model: Modelo entrenado
        feature_names: Nombres de features
        title: Título
        top_n: Top N features a mostrar
        save_path:  Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("⚠️ Modelo no tiene feature_importances_")
        return None
    
    # Crear DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(importance_df['feature'], importance_df['importance'], edgecolor='black')
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {save_path}")
    
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Error Distribution",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Distribución de errores (absolutos y porcentuales)
    
    Args:
        y_true: Valores reales
        y_pred:  Predicciones
        title:  Título
        save_path:  Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    absolute_errors = np.abs(y_true - y_pred)
    percentage_errors = (absolute_errors / np.abs(y_true)) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute errors
    axes[0].hist(absolute_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=absolute_errors.mean(), color='r', linestyle='--', 
                    linewidth=2, label=f'Mean: {absolute_errors.mean():.2f}')
    axes[0].set_xlabel('Absolute Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Absolute Errors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Percentage errors
    axes[1].hist(percentage_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(x=percentage_errors.mean(), color='r', linestyle='--', 
                    linewidth=2, label=f'Mean: {percentage_errors.mean():.2f}%')
    axes[1].set_xlabel('Percentage Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{title} - Percentage Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {save_path}")
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'test_rmse',
    title: str = "Model Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Comparación visual de modelos
    
    Args:
        comparison_df: DataFrame con comparación
        metric: Métrica a visualizar
        title: Título
        save_path: Ruta para guardar
        
    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = comparison_df['model']
    values = comparison_df[metric]
    
    bars = ax.bar(models, values, edgecolor='black', alpha=0.7)
    
    # Colorear mejor modelo
    if 'r2' in metric.lower():
        best_idx = values.idxmax()
    else:
        best_idx = values.idxmin()
    
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)
    
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Añadir valores en las barras
    for i, (model, value) in enumerate(zip(models, values)):
        ax.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {save_path}")
    
    return fig