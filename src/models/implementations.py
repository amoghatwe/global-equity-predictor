"""
Machine Learning Model Implementations for Equity Return Prediction
===================================================================

This module contains the actual model implementations used to predict
3-year forward equity returns. Each model wraps a scikit-learn or
statsmodels estimator and provides a consistent interface for training
and prediction.

Models included:
    - LinearReturnModel: Ridge regression (interpretable baseline)
    - RandomForestReturnModel: Non-linear ensemble of decision trees
    - XGBoostReturnModel: Gradient boosting with regularization
    - ARMAReturnModel: ARMA time series model with exogenous features
    - ARIMAReturnModel: ARIMA with differencing for non-stationary series

All models inherit from BaseReturnModel which defines the common interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
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
    Linear regression model with Ridge (L2) regularization.

    Why Ridge Regression?
    ---------------------
    Ridge regression is a linear model that adds a penalty term to the loss
    function. This penalty shrinks the coefficients, which helps prevent
    overfitting when you have many correlated features (common in financial
    data where indicators like GDP growth and industrial production often
    move together).

    The 'alpha' parameter controls regularization strength:
    - High alpha = more regularization = simpler model
    - Low alpha = less regularization = more complex model

    This model serves as an interpretable baseline. If complex models like
    XGBoost can't beat this, you probably don't need the complexity.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Ridge regression model.

        Args:
            **kwargs: Override default parameters (e.g., alpha=0.5)
        """
        # Merge default parameters with any user-provided overrides
        # This lets you tweak settings without changing the defaults
        params = {**LINEAR_PARAMS, **kwargs}
        model = Ridge(**params)
        super().__init__("LinearRegression", model)

    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Train the Ridge regression model on historical data.

        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target vector (3-year forward returns)

        Returns:
            Dictionary with training metrics (RMSE, R², etc.)

        Raises:
            ValueError: If no valid training samples remain after cleaning
        """
        self.logger.info(
            f"Training {self.name} on {len(X)} samples, {X.shape[1]} features..."
        )

        # Store feature names for later interpretation
        # We need these to explain which features drive predictions
        self.feature_names = list(X.columns)
        self.target_name = y.name

        # Remove rows where target is NaN
        # This happens for recent dates where we don't have 3-year forward data yet
        valid_target_mask = y.notna()
        X_valid = X[valid_target_mask]
        y_valid = y[valid_target_mask]

        if len(X_valid) == 0:
            raise ValueError(
                "No valid training samples found. "
                "Check that your target variable has non-NaN values."
            )

        # Handle missing values in features by filling with median
        # Median is robust to outliers (unlike mean)
        X_filled = X_valid.fillna(X_valid.median())

        # Fit the model
        self.model.fit(X_filled, y_valid)
        self.is_trained = True

        # Evaluate on training data to check for overfitting
        y_predictions = self.model.predict(X_filled)

        # Calculate standard regression metrics
        metrics = self.evaluate(y_valid.values, y_predictions)
        metrics["n_samples"] = len(X_valid)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, "
            f"R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data.

        Args:
            X: Feature matrix with same columns as training data

        Returns:
            Array of predicted 3-year forward returns

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError(
                "Cannot predict: model has not been trained. "
                "Call train() first."
            )

        # Handle missing values same way as training
        X_filled = X.fillna(X.median())

        return self.model.predict(X_filled)

    def get_coefficients(self) -> pd.Series:
        """
        Get model coefficients to interpret feature importance.

        Returns:
            Series of coefficients sorted by absolute magnitude

        Interpretation:
            - Positive coefficient: feature increases predicted returns
            - Negative coefficient: feature decreases predicted returns
            - Larger magnitude = stronger effect
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting coefficients")

        coefficients = pd.Series(
            self.model.coef_, index=self.feature_names
        ).sort_values(key=abs, ascending=False)

        return coefficients


class RandomForestReturnModel(BaseReturnModel):
    """
    Random Forest regression model for capturing non-linear relationships.

    What is a Random Forest?
    ------------------------
    A Random Forest is an ensemble of decision trees. Each tree is trained on
    a random subset of the data and features, then predictions are averaged
    across all trees.

    Why use Random Forest?
    ----------------------
    - Captures non-linear relationships (e.g., valuation matters more at extremes)
    - Handles feature interactions automatically (e.g., high inflation + low growth)
    - Provides feature importance scores
    - Less prone to overfitting than a single decision tree

    Key parameters:
    - n_estimators: Number of trees (more = better but slower)
    - max_depth: Maximum tree depth (limits complexity)
    - min_samples_split: Minimum samples to split a node (prevents overfitting)
    """

    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model.

        Args:
            **kwargs: Override default parameters
        """
        params = {**RF_PARAMS, **kwargs}
        model = RandomForestRegressor(**params)
        super().__init__("RandomForest", model)

    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix
            y: Target vector (3-year forward returns)

        Returns:
            Training metrics including RMSE and R²
        """
        self.logger.info(
            f"Training {self.name} on {len(X)} samples, {X.shape[1]} features..."
        )

        self.feature_names = list(X.columns)
        self.target_name = y.name

        # Remove samples with missing target values
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Fill missing feature values with median (robust to outliers)
        X_filled = X_valid.fillna(X_valid.median())

        # Train the forest
        self.model.fit(X_filled, y_valid)
        self.is_trained = True

        # Evaluate on training set
        y_predictions = self.model.predict(X_filled)
        metrics = self.evaluate(y_valid.values, y_predictions)
        metrics["n_samples"] = len(X_valid)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, "
            f"R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained forest.

        Each tree votes, and we average the results.
        """
        if not self.is_trained:
            raise ValueError(
                "Cannot predict: model has not been trained. "
                "Call train() first."
            )

        # Handle missing values consistently with training
        X_filled = X.fillna(X.median())
        return self.model.predict(X_filled)


class XGBoostReturnModel(BaseReturnModel):
    """
    XGBoost (Extreme Gradient Boosting) regression model.

    What is XGBoost?
    ----------------
    XGBoost builds trees sequentially, where each new tree corrects the errors
    of the previous ones. Think of it as:
    1. Train a weak learner (slightly better than random guessing)
    2. See where it fails
    3. Train next learner to fix those mistakes
    4. Repeat hundreds of times

    Why XGBoost?
    ------------
    - Often the best performing off-the-shelf ML model
    - Handles non-linear relationships and feature interactions
    - Built-in regularization prevents overfitting
    - Provides feature importance scores

    Key regularization parameters:
    - max_depth: Controls tree complexity (deeper = more complex)
    - learning_rate: How much each tree contributes (lower = more trees needed)
    - reg_alpha/reg_lambda: L1/L2 regularization on weights
    - subsample/colsample_bytree: Random sampling for diversity

    Tip: XGBoost is powerful but can overfit. Always compare to simpler models.
    """

    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model.

        Args:
            **kwargs: Override default parameters
        """
        params = {**XGB_PARAMS, **kwargs}
        model = XGBRegressor(**params)
        super().__init__("XGBoost", model)

    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            X: Feature matrix
            y: Target vector (3-year forward returns)

        Returns:
            Training metrics including RMSE and R²
        """
        self.logger.info(
            f"Training {self.name} on {len(X)} samples, {X.shape[1]} features..."
        )

        self.feature_names = list(X.columns)
        self.target_name = y.name

        # Remove samples with missing target values
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Fill missing values - XGBoost handles NaN internally, but we fill
        # to be consistent across models and avoid surprises
        X_filled = X_valid.fillna(X_valid.median())

        # Train the gradient boosted trees
        self.model.fit(X_filled, y_valid)
        self.is_trained = True

        # Evaluate performance on training data
        y_predictions = self.model.predict(X_filled)
        metrics = self.evaluate(y_valid.values, y_predictions)
        metrics["n_samples"] = len(X_valid)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, "
            f"R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted returns
        """
        if not self.is_trained:
            raise ValueError(
                "Cannot predict: model has not been trained. "
                "Call train() first."
            )

        # Handle missing values consistently
        X_filled = X.fillna(X.median())
        return self.model.predict(X_filled)

    def get_feature_importance_plot(self, top_n: int = 20) -> None:
        """
        Display a horizontal bar chart of the most important features.

        Args:
            top_n: Number of top features to display (default: 20)

        Note: Requires matplotlib. If not available, silently skips plotting.
        """
        try:
            import matplotlib.pyplot as plt

            importance_scores = self.get_feature_importance()
            if importance_scores is not None:
                plt.figure(figsize=(10, 8))
                importance_scores.head(top_n).plot(kind="barh")
                plt.title(f"Top {top_n} Feature Importances - {self.name}")
                plt.tight_layout()
                plt.show()
        except ImportError:
            self.logger.warning(
                "matplotlib not installed - skipping feature importance plot"
            )


class ARMAReturnModel(BaseReturnModel):
    """
    ARMA (AutoRegressive Moving Average) model for time series forecasting.

    What is ARMA?
    -------------
    ARMA models predict future values based on:
    - AR (Autoregressive) part: Past values of the series itself
    - MA (Moving Average) part: Past forecast errors

    An ARMA(p, q) model uses:
    - p past values (autoregressive terms)
    - q past forecast errors (moving average terms)

    Why ARMA for equity returns?
    ----------------------------
    - Captures momentum and mean-reversion patterns
    - Models serial correlation in returns
    - Can include exogenous variables (then called ARMAX)
    - Useful baseline for time series forecasting

    Order selection guidance:
    - Start simple: ARMA(1,1) often works well
    - Higher p = more memory of past values
    - Higher q = more complex error correction
    - Use AIC/BIC to compare different orders

    Note: ARMA assumes stationarity (constant mean/variance over time).
    If your series has a trend, consider ARIMA which adds differencing.

    Parameters:
    - order: Tuple (p, q) specifying AR and MA orders
    - trend: 'c' for constant, 'ct' for constant+trend, 'n' for none
    """

    def __init__(self, **kwargs):
        """
        Initialize the ARMA model.

        Args:
            **kwargs: Override default parameters
                order: Tuple (p, q) - default (1, 1)
                trend: 'c', 'ct', or 'n' - default 'c'
        """
        params = {**ARMA_PARAMS, **kwargs}
        self.arma_order = params.pop("order", (1, 1))
        self.arma_trend = params.pop("trend", "c")
        super().__init__("ARMA", None)

    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Fit the ARMA model to historical data.

        The model learns:
        1. How past returns predict future returns (AR part)
        2. How to correct for past forecast errors (MA part)
        3. How exogenous features (like valuation, inflation) affect returns

        Args:
            X: Exogenous features (optional, becomes ARMAX if provided)
            y: Target series (3-year forward returns)

        Returns:
            Training metrics including RMSE and R²

        Raises:
            ValueError: If insufficient data for the specified ARMA order
        """
        # Import here to avoid requiring statsmodels unless ARMA is used
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.logger.info(
            f"Training {self.name}({self.arma_order}) on {len(X)} samples, "
            f"{X.shape[1]} exogenous features..."
        )

        self.feature_names = list(X.columns)
        self.target_name = y.name

        # Remove samples with missing target values
        valid_mask = y.notna()
        X_valid = X[valid_mask].fillna(X[valid_mask].median())
        y_valid = y[valid_mask]

        # ARMA needs enough data relative to the order
        # Rule of thumb: at least 3x the order plus some buffer
        minimum_required_samples = max(self.arma_order) * 3 + 10
        if len(X_valid) < minimum_required_samples:
            raise ValueError(
                f"Insufficient data for ARMA{self.arma_order}. "
                f"Have {len(X_valid)} samples, need at least {minimum_required_samples}. "
                f"Try a simpler model (lower order) or collect more data."
            )

        # Build the SARIMAX model
        # We use SARIMAX because it supports exogenous variables
        # The 'S' and 'I' parts are disabled (order has d=0, no seasonal terms)
        time_series_model = SARIMAX(
            endog=y_valid,  # The series we're predicting
            exog=X_valid if X_valid.shape[1] > 0 else None,  # Optional features
            order=(
                self.arma_order[0],  # p = AR order
                0,  # d = differencing (0 for ARMA, >0 for ARIMA)
                self.arma_order[1],  # q = MA order
            ),
            trend=self.arma_trend,
            # Don't enforce stationarity/invertibility - let the data decide
            # This allows the model to fit even if the series is borderline
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        # Fit the model using maximum likelihood estimation
        # disp=False suppresses the optimization output
        self.model = time_series_model.fit(disp=False, maxiter=200)
        self.is_trained = True

        # Get fitted values (in-sample predictions)
        y_fitted = self.model.fittedvalues.values
        y_actual = y_valid.values

        # Align lengths (fitted values may be shorter due to AR/MA lags)
        alignment_length = min(len(y_actual), len(y_fitted))
        metrics = self.evaluate(
            y_actual[-alignment_length:], y_fitted[-alignment_length:]
        )
        metrics["n_samples"] = len(X_valid)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, "
            f"R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasts using the fitted ARMA model.

        Args:
            X: Exogenous features for the forecast period

        Returns:
            Array of predicted 3-year forward returns

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError(
                "Cannot predict: ARMA model has not been trained. "
                "Call train() first."
            )

        # Handle missing values
        X_filled = X.fillna(X.median())
        forecast_steps = len(X_filled)

        # Use exogenous variables only if provided
        exog_features = X_filled if X_filled.shape[1] > 0 else None

        # Get forecast object which contains predictions and confidence intervals
        forecast_result = self.model.get_forecast(
            steps=forecast_steps, exog=exog_features
        )
        return forecast_result.predicted_mean.values

    def save(self, filepath: str):
        """
        Save the trained ARMA model to disk.

        We use pickle because statsmodels models don't have a native save format.

        Args:
            filepath: Path to save the model (should end in .pkl)
        """
        import pickle

        with open(filepath, "wb") as file:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "target_name": self.target_name,
                    "is_trained": self.is_trained,
                    "arma_order": self.arma_order,
                    "arma_trend": self.arma_trend,
                },
                file,
            )
        self.logger.info(f"Saved ARMA model to {filepath}")

    def load(self, filepath: str):
        """
        Load a previously saved ARMA model.

        Args:
            filepath: Path to the saved model file
        """
        import pickle

        with open(filepath, "rb") as file:
            data = pickle.load(file)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.target_name = data["target_name"]
            self.is_trained = data["is_trained"]
            self.arma_order = data.get("arma_order", (1, 1))
            self.arma_trend = data.get("arma_trend", "c")
        self.logger.info(f"Loaded ARMA model from {filepath}")


class ARIMAReturnModel(BaseReturnModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for non-stationary series.

    What is ARIMA?
    --------------
    ARIMA extends ARMA by adding differencing, which makes it suitable for
    series that aren't stationary (i.e., their statistical properties change
    over time).

    ARIMA(p, d, q) components:
    - p (AR): Past values influence the future
    - d (I): Differencing order - how many times we subtract past values
    - q (MA): Past forecast errors influence predictions

    What is differencing?
    ---------------------
    Differencing transforms a non-stationary series into a stationary one by
    computing changes between consecutive observations:
    - First difference: y(t) - y(t-1)  [removes trend]
    - Second difference: [y(t) - y(t-1)] - [y(t-1) - y(t-2)]  [removes curvature]

    When to use ARIMA vs ARMA?
    --------------------------
    - Use ARMA if the series looks stationary (constant mean/variance)
    - Use ARIMA if the series has a clear trend or changing variance
    - For equity returns: ARMA often suffices since returns are already
      stationary (unlike prices)

    Parameter guidance:
    - d=0: Series is already stationary (equivalent to ARMA)
    - d=1: Series has a linear trend (most common)
    - d=2: Series has changing growth rate (rarely needed)

    Tip: Use statistical tests (ADF, KPSS) to determine if differencing is needed.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ARIMA model.

        Args:
            **kwargs: Override default parameters
                order: Tuple (p, d, q) - default (1, 1, 1)
                trend: 'c', 'ct', or 'n' - default 'c'
        """
        params = {**ARIMA_PARAMS, **kwargs}
        self.arima_order = params.pop("order", (1, 1, 1))
        self.arima_trend = params.pop("trend", "c")
        super().__init__("ARIMA", None)

    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Fit the ARIMA model to historical data.

        The differencing parameter (d) automatically handles non-stationarity
        by computing changes between observations before fitting the ARMA part.

        Args:
            X: Exogenous features (optional, becomes ARIMAX if provided)
            y: Target series (3-year forward returns)

        Returns:
            Training metrics including RMSE and R²

        Raises:
            ValueError: If insufficient data for the specified ARIMA order
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.logger.info(
            f"Training {self.name}({self.arima_order}) on {len(X)} samples, "
            f"{X.shape[1]} exogenous features..."
        )

        self.feature_names = list(X.columns)
        self.target_name = y.name

        # Remove samples with missing target values
        valid_mask = y.notna()
        X_valid = X[valid_mask].fillna(X[valid_mask].median())
        y_valid = y[valid_mask]

        # ARIMA needs sufficient data relative to the order
        # The differencing operation reduces effective sample size
        minimum_required_samples = max(self.arima_order) * 3 + 10
        if len(X_valid) < minimum_required_samples:
            raise ValueError(
                f"Insufficient data for ARIMA{self.arima_order}. "
                f"Have {len(X_valid)} samples, need at least {minimum_required_samples}. "
                f"Try a simpler model (lower order) or collect more data."
            )

        # Build the SARIMAX model with full ARIMA specification
        # SARIMAX handles the differencing internally based on the order parameter
        time_series_model = SARIMAX(
            endog=y_valid,  # The series we're predicting
            exog=X_valid if X_valid.shape[1] > 0 else None,  # Optional features
            order=self.arima_order,  # (p, d, q) - includes differencing
            trend=self.arima_trend,
            # Relax constraints to allow fitting borderline cases
            # The model will warn if parameters are problematic
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        # Fit using maximum likelihood estimation
        # This finds parameters that make the observed data most probable
        self.model = time_series_model.fit(disp=False, maxiter=200)
        self.is_trained = True

        # Get in-sample fitted values
        y_fitted = self.model.fittedvalues.values
        y_actual = y_valid.values

        # Align lengths (fitted values shorter due to AR/MA lags and differencing)
        alignment_length = min(len(y_actual), len(y_fitted))
        metrics = self.evaluate(
            y_actual[-alignment_length:], y_fitted[-alignment_length:]
        )
        metrics["n_samples"] = len(X_valid)
        metrics["n_features"] = X.shape[1]

        self.logger.info(
            f"Training completed. RMSE: {metrics['rmse']:.2f}, "
            f"R²: {metrics['r2']:.3f}"
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasts using the fitted ARIMA model.

        The model automatically handles the integration (differencing reversal)
        to produce predictions in the original scale.

        Args:
            X: Exogenous features for the forecast period

        Returns:
            Array of predicted 3-year forward returns

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError(
                "Cannot predict: ARIMA model has not been trained. "
                "Call train() first."
            )

        # Handle missing values consistently with training
        X_filled = X.fillna(X.median())
        forecast_steps = len(X_filled)

        # Include exogenous variables only if we have features
        exog_features = X_filled if X_filled.shape[1] > 0 else None

        # Get forecast with predictions and confidence intervals
        # The model automatically reverses the differencing transformation
        forecast_result = self.model.get_forecast(
            steps=forecast_steps, exog=exog_features
        )
        return forecast_result.predicted_mean.values

    def save(self, filepath: str):
        """
        Save the trained ARIMA model to disk.

        Args:
            filepath: Path to save the model (.pkl extension recommended)
        """
        import pickle

        with open(filepath, "wb") as file:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "target_name": self.target_name,
                    "is_trained": self.is_trained,
                    "arima_order": self.arima_order,
                    "arima_trend": self.arima_trend,
                },
                file,
            )
        self.logger.info(f"Saved ARIMA model to {filepath}")

    def load(self, filepath: str):
        """
        Load a previously saved ARIMA model.

        Args:
            filepath: Path to the saved model file
        """
        import pickle

        with open(filepath, "rb") as file:
            data = pickle.load(file)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.target_name = data["target_name"]
            self.is_trained = data["is_trained"]
            self.arima_order = data.get("arima_order", (1, 1, 1))
            self.arima_trend = data.get("arima_trend", "c")
        self.logger.info(f"Loaded ARIMA model from {filepath}")


def create_model(model_type: str, **kwargs) -> BaseReturnModel:
    """
    Factory function to create model instances by name.

    Why use a factory function?
    ---------------------------
    Instead of remembering which class to import, you just pass a string:
    - create_model("linear") instead of LinearReturnModel()
    - create_model("arma") instead of ARMAReturnModel()

    This is especially useful in configuration files and CLI tools.

    Supported model types:
    ----------------------
    Traditional ML:
    - "linear" or "ridge": Ridge regression (interpretable baseline)
    - "rf" or "random_forest": Random Forest (non-linear, robust)
    - "xgb" or "xgboost": XGBoost (gradient boosting, often best performance)

    Time Series:
    - "arma": ARMA for stationary series with exogenous features
    - "arima": ARIMA for non-stationary series (adds differencing)

    Args:
        model_type: String identifier for the model (case-sensitive)
        **kwargs: Model-specific parameters passed to the constructor

    Returns:
        An instance of the requested model class, ready to train

    Raises:
        ValueError: If model_type is not recognized

    Example:
        >>> model = create_model("arma", order=(2, 1))
        >>> metrics = model.train(X_train, y_train)
    """
    # Map of model identifiers to their classes
    # Multiple names for the same model for user convenience
    model_classes = {
        # Traditional machine learning models
        "linear": LinearReturnModel,
        "ridge": LinearReturnModel,  # Alias for linear
        "rf": RandomForestReturnModel,
        "random_forest": RandomForestReturnModel,  # Full name alias
        "xgb": XGBoostReturnModel,
        "xgboost": XGBoostReturnModel,  # Full name alias
        # Time series models
        "arma": ARMAReturnModel,
        "arima": ARIMAReturnModel,
    }

    # Check if the requested model type exists
    if model_type not in model_classes:
        available_models = sorted(model_classes.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'.\n"
            f"Available models: {', '.join(available_models)}\n"
            f"\n"
            f"Examples:\n"
            f"  create_model('linear')  - Simple, interpretable baseline\n"
            f"  create_model('arma')    - Time series with momentum\n"
            f"  create_model('xgboost') - Powerful gradient boosting"
        )

    # Instantiate and return the model
    return model_classes[model_type](**kwargs)
