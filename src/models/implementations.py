"""
Model implementations: Linear Regression, Random Forest, XGBoost, ARMA, ARIMA.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .base import BaseReturnModel
from config.settings import (
    LINEAR_PARAMS,
    RF_PARAMS,
    XGB_PARAMS,
    ARMA_PARAMS,
    ARIMA_PARAMS,
)

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
        self.logger.info(
            f"Training {self.name} on {len(X)} samples, {X.shape[1]} features..."
        )

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
        metrics["n_samples"] = len(X_clean)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )

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

        coefs = pd.Series(self.model.coef_, index=self.feature_names).sort_values(
            key=abs, ascending=False
        )

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
        self.logger.info(
            f"Training {self.name} on {len(X)} samples, {X.shape[1]} features..."
        )

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
        metrics["n_samples"] = len(X_clean)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )

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
        self.logger.info(
            f"Training {self.name} on {len(X)} samples, {X.shape[1]} features..."
        )

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
        metrics["n_samples"] = len(X_clean)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )

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
                importance.head(top_n).plot(kind="barh")
                plt.title(f"Top {top_n} Feature Importances - {self.name}")
                plt.tight_layout()
                plt.show()
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")


class ARMAReturnModel(BaseReturnModel):
    """
    ARMA (AutoRegressive Moving Average) model using statsmodels SARIMAX.

    Fits ARMA(p,q) on the target series with optional exogenous features.
    The ARMA order (p,q) is configurable; differencing d=0 (no integration).
    """

    def __init__(self, **kwargs):
        params = {**ARMA_PARAMS, **kwargs}
        self.arma_order = params.pop("order", (1, 1))
        self.arma_trend = params.pop("trend", "c")
        super().__init__("ARMA", None)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.logger.info(
            f"Training {self.name}({self.arma_order}) on {len(X)} samples, {X.shape[1]} exog features..."
        )

        self.feature_names = list(X.columns)
        self.target_name = y.name

        mask = y.notna()
        X_clean = X[mask].fillna(X[mask].median())
        y_clean = y[mask]

        if len(X_clean) < max(self.arma_order) * 3 + 10:
            raise ValueError(
                f"Not enough samples ({len(X_clean)}) for ARMA{self.arma_order}"
            )

        model = SARIMAX(
            endog=y_clean,
            exog=X_clean if X_clean.shape[1] > 0 else None,
            order=(self.arma_order[0], 0, self.arma_order[1]),
            trend=self.arma_trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.model = model.fit(disp=False, maxiter=200)
        self.is_trained = True

        y_pred = self.model.fittedvalues.values
        y_true = y_clean.values

        min_len = min(len(y_true), len(y_pred))
        metrics = self.evaluate(y_true[-min_len:], y_pred[-min_len:])
        metrics["n_samples"] = len(X_clean)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_filled = X.fillna(X.median())
        n_steps = len(X_filled)

        exog = X_filled if X_filled.shape[1] > 0 else None
        forecast = self.model.get_forecast(steps=n_steps, exog=exog)
        return forecast.predicted_mean.values

    def save(self, filepath: str):
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "target_name": self.target_name,
                    "is_trained": self.is_trained,
                    "arma_order": self.arma_order,
                    "arma_trend": self.arma_trend,
                },
                f,
            )
        self.logger.info(f"Saved model to {filepath}")

    def load(self, filepath: str):
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.target_name = data["target_name"]
            self.is_trained = data["is_trained"]
            self.arma_order = data.get("arma_order", (1, 1))
            self.arma_trend = data.get("arma_trend", "c")
        self.logger.info(f"Loaded model from {filepath}")


class ARIMAReturnModel(BaseReturnModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model using statsmodels SARIMAX.

    Fits ARIMA(p,d,q) on the target series with optional exogenous features.
    The differencing order d enables modeling of non-stationary series.
    """

    def __init__(self, **kwargs):
        params = {**ARIMA_PARAMS, **kwargs}
        self.arima_order = params.pop("order", (1, 1, 1))
        self.arima_trend = params.pop("trend", "c")
        super().__init__("ARIMA", None)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.logger.info(
            f"Training {self.name}({self.arima_order}) on {len(X)} samples, {X.shape[1]} exog features..."
        )

        self.feature_names = list(X.columns)
        self.target_name = y.name

        mask = y.notna()
        X_clean = X[mask].fillna(X[mask].median())
        y_clean = y[mask]

        if len(X_clean) < max(self.arima_order) * 3 + 10:
            raise ValueError(
                f"Not enough samples ({len(X_clean)}) for ARIMA{self.arima_order}"
            )

        model = SARIMAX(
            endog=y_clean,
            exog=X_clean if X_clean.shape[1] > 0 else None,
            order=self.arima_order,
            trend=self.arima_trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.model = model.fit(disp=False, maxiter=200)
        self.is_trained = True

        y_pred = self.model.fittedvalues.values
        y_true = y_clean.values

        min_len = min(len(y_true), len(y_pred))
        metrics = self.evaluate(y_true[-min_len:], y_pred[-min_len:])
        metrics["n_samples"] = len(X_clean)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")

        X_filled = X.fillna(X.median())
        n_steps = len(X_filled)

        exog = X_filled if X_filled.shape[1] > 0 else None
        forecast = self.model.get_forecast(steps=n_steps, exog=exog)
        return forecast.predicted_mean.values

    def save(self, filepath: str):
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "target_name": self.target_name,
                    "is_trained": self.is_trained,
                    "arima_order": self.arima_order,
                    "arima_trend": self.arima_trend,
                },
                f,
            )
        self.logger.info(f"Saved model to {filepath}")

    def load(self, filepath: str):
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.target_name = data["target_name"]
            self.is_trained = data["is_trained"]
            self.arima_order = data.get("arima_order", (1, 1, 1))
            self.arima_trend = data.get("arima_trend", "c")
        self.logger.info(f"Loaded model from {filepath}")


def create_model(model_type: str, **kwargs) -> BaseReturnModel:
    """
    Factory function to create model instances.

    Args:
        model_type: 'linear', 'rf', 'xgb', 'arma', or 'arima'
        **kwargs: Model-specific parameters

    Returns:
        Model instance
    """
    model_map = {
        "linear": LinearReturnModel,
        "rf": RandomForestReturnModel,
        "xgb": XGBoostReturnModel,
        "arma": ARMAReturnModel,
        "arima": ARIMAReturnModel,
        "ridge": LinearReturnModel,
        "random_forest": RandomForestReturnModel,
        "xgboost": XGBoostReturnModel,
    }

    if model_type not in model_map:
        raise ValueError(
            f"Unknown model type: {model_type}. Use: {list(model_map.keys())}"
        )

    return model_map[model_type](**kwargs)
