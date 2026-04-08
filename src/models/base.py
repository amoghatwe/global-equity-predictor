"""
Base classes and utilities for machine learning models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """
    Time series aware cross-validation with expanding and rolling windows.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        min_train_years: int = 15,
        max_train_years: Optional[int] = None,
        gap_months: int = 0,
        embargo_pct: float = 0.0
    ):
        """
        Initialize time series CV.
        
        Args:
            n_splits: Number of train/test splits
            min_train_years: Minimum training period in years
            max_train_years: Maximum training period (None for expanding)
            gap_months: Gap between train and test to avoid leakage
            embargo_pct: Percentage of test set to embargo (for overlapping returns)
        """
        self.n_splits = n_splits
        self.min_train_months = min_train_years * 12
        self.max_train_months = max_train_years * 12 if max_train_years else None
        self.gap_months = gap_months
        self.embargo_pct = embargo_pct
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Generate train/test splits for time series.
        
        Args:
            X: Feature matrix (sorted by time)
            y: Target variable (not used, for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate fold size
        available_for_test = n_samples - self.min_train_months - self.gap_months
        fold_size = available_for_test // self.n_splits
        
        if fold_size < 12:  # At least 1 year of test data
            raise ValueError(f"Not enough data for {self.n_splits} splits. "
                           f"Need at least {self.min_train_months + self.n_splits * 12} months")
        
        for i in range(self.n_splits):
            # Test period
            test_start = self.min_train_months + i * fold_size
            test_end = min(test_start + fold_size, n_samples)
            
            # Embargo (remove beginning of test set to avoid overlap with train)
            embargo = int(fold_size * self.embargo_pct)
            test_start += embargo
            
            # Train period
            if self.max_train_months:
                # Rolling window
                train_start = max(0, test_start - self.max_train_months)
            else:
                # Expanding window
                train_start = 0
            
            train_end = test_start - self.gap_months
            
            if train_end <= train_start:
                continue
            
            yield (
                indices[train_start:train_end],
                indices[test_start:test_end]
            )
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


class BaseReturnModel(ABC):
    """
    Abstract base class for return prediction models.
    """
    
    def __init__(self, name: str, model: BaseEstimator):
        self.name = name
        self.model = model
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dict with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        pass
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'directional_accuracy': np.nan,
                'correlation': np.nan
            }
        
        # Standard metrics
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        # Directional accuracy
        direction_true = np.sign(y_true_clean)
        direction_pred = np.sign(y_pred_clean)
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        # Correlation
        correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation
        }
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
            Series with feature importances or None
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            return None
        
        if self.feature_names:
            return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)
    
    def save(self, filepath: str):
        """Save model to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'is_trained': self.is_trained
            }, f)
        self.logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.target_name = data['target_name']
            self.is_trained = data['is_trained']
        self.logger.info(f"Loaded model from {filepath}")


class ModelEnsemble:
    """
    Ensemble of multiple models.
    """
    
    def __init__(self, models: Dict[str, BaseReturnModel], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of model_name: model_instance
            weights: Optional weights for each model (default: equal weighting)
        """
        self.models = models
        self.weights = weights or {name: 1.0 for name in models}
        self.logger = logging.getLogger(__name__)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights.get(name, 1.0))
        
        if not predictions:
            raise ValueError("No trained models in ensemble")
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models and ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of metrics for each model and ensemble
        """
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            if model.is_trained:
                pred = model.predict(X)
                results[name] = model.evaluate(y.values, pred)
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X)
        base_model = list(self.models.values())[0]
        results['ensemble'] = base_model.evaluate(y.values, ensemble_pred)
        
        return results
