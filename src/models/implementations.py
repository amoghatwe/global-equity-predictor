"""
Model implementations: Linear Regression, Random Forest, XGBoost.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .base import BaseReturnModel
from config.settings import LINEAR_PARAMS, RF_PARAMS, XGB_PARAMS

logger = logging.getLogger(__name__)


class LinearReturnModel(BaseReturnModel):
    """
    Linear regression model with Ridge regularization.
    
    Simple, interpretable baseline model.
    """
    
    def __init__(self, **kwargs):
        # Merge default params with any overrides
        params = {**LINEAR_PARAMS, **kwargs}
        model = Ridge(**params)
        super().__init__("LinearRegression", model)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the linear model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Training {self.name} on {len(X)} samples, {X.shape[1]} features...")
        
        # Store feature and target names
        self.feature_names = list(X.columns)
        self.target_name = y.name
        
        # Remove rows with NaN in target
        mask = y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid training samples")
        
        # Handle missing values in features
        X_clean = X_clean.fillna(X_clean.median())
        
        # Train
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        # Get predictions on training set
        y_pred = self.model.predict(X_clean)
        
        # Calculate metrics
        metrics = self.evaluate(y_clean.values, y_pred)
        metrics['n_samples'] = len(X_clean)
        metrics['n_features'] = X.shape[1]
        
        self.logger.info(f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Handle missing values
        X_filled = X.fillna(X.median())
        
        return self.model.predict(X_filled)
    
    def get_coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        coefs = pd.Series(
            self.model.coef_,
            index=self.feature_names
        ).sort_values(key=abs, ascending=False)
        
        return coefs


class RandomForestReturnModel(BaseReturnModel):
    """
    Random Forest regression model.
    
    Non-linear model with feature importance.
    """
    
    def __init__(self, **kwargs):
        params = {**RF_PARAMS, **kwargs}
        model = RandomForestRegressor(**params)
        super().__init__("RandomForest", model)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Training {self.name} on {len(X)} samples, {X.shape[1]} features...")
        
        self.feature_names = list(X.columns)
        self.target_name = y.name
        
        # Remove NaN
        mask = y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Handle missing values
        X_clean = X_clean.fillna(X_clean.median())
        
        # Train
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        # Metrics
        y_pred = self.model.predict(X_clean)
        metrics = self.evaluate(y_clean.values, y_pred)
        metrics['n_samples'] = len(X_clean)
        metrics['n_features'] = X.shape[1]
        
        self.logger.info(f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_filled = X.fillna(X.median())
        return self.model.predict(X_filled)


class XGBoostReturnModel(BaseReturnModel):
    """
    XGBoost regression model.
    
    Gradient boosting with regularization.
    """
    
    def __init__(self, **kwargs):
        params = {**XGB_PARAMS, **kwargs}
        model = XGBRegressor(**params)
        super().__init__("XGBoost", model)
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Training {self.name} on {len(X)} samples, {X.shape[1]} features...")
        
        self.feature_names = list(X.columns)
        self.target_name = y.name
        
        # Remove NaN
        mask = y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Handle missing values (XGBoost can handle NaN, but let's be safe)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Train
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        # Metrics
        y_pred = self.model.predict(X_clean)
        metrics = self.evaluate(y_clean.values, y_pred)
        metrics['n_samples'] = len(X_clean)
        metrics['n_features'] = X.shape[1]
        
        self.logger.info(f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_filled = X.fillna(X.median())
        return self.model.predict(X_filled)
    
    def get_feature_importance_plot(self, top_n: int = 20) -> None:
        """Plot feature importance."""
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance()
            if importance is not None:
                plt.figure(figsize=(10, 8))
                importance.head(top_n).plot(kind='barh')
                plt.title(f'Top {top_n} Feature Importances - {self.name}')
                plt.tight_layout()
                plt.show()
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")


def create_model(model_type: str, **kwargs) -> BaseReturnModel:
    """
    Factory function to create model instances.
    
    Args:
        model_type: 'linear', 'rf', or 'xgb'
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    model_map = {
        'linear': LinearReturnModel,
        'rf': RandomForestReturnModel,
        'xgb': XGBoostReturnModel,
        'ridge': LinearReturnModel,
        'random_forest': RandomForestReturnModel,
        'xgboost': XGBoostReturnModel,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Use: {list(model_map.keys())}")
    
    return model_map[model_type](**kwargs)
