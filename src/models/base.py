"""
Base Classes and Utilities for Machine Learning Models
======================================================

This module provides the foundational classes that all return prediction models
build upon. Think of this as the "plumbing" that makes sure everything works
consistently across different model types.

Key components:
    - TimeSeriesCrossValidator: Proper time series splits (no data leakage!)
    - BaseReturnModel: Common interface all models must follow
    - ModelEnsemble: Combine multiple models for better predictions

Why time series cross-validation?
---------------------------------
Standard k-fold CV randomly shuffles data, which is a big no-no for time series.
You can't use tomorrow's data to predict yesterday! This validator ensures:
    1. Training data always comes before test data (chronologically)
    2. No look-ahead bias (using future information)
    3. Realistic evaluation of how the model would perform in production
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
    Cross-validation that respects the arrow of time.

    Why is this necessary?
    ----------------------
    In standard machine learning, you can randomly shuffle data because each
    sample is independent. But with time series:
    - Today's stock return depends on yesterday's information
    - You can't use future data to predict the past (that's cheating!)
    - Standard CV would give overly optimistic results

    This validator creates splits where training data ALWAYS comes before
    test data, simulating how you'd actually use the model in real life.

    Window types:
    -------------
    - Expanding window: Training set grows over time (more data = better)
    - Rolling window: Training set stays fixed size (adapts to regime changes)

    Embargo period:
    ---------------
    For overlapping targets (e.g., 3-year forward returns), consecutive
    samples share information. The embargo removes some test samples to
    prevent this leakage.
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
        Initialize the time series cross-validator.

        Args:
            n_splits: How many train/test splits to create (like k in k-fold)
            min_train_years: Minimum years of data needed before first test split
                - More years = more reliable models but fewer test splits
                - 10-15 years is typical for financial data
            max_train_years: Maximum training window (None for expanding)
                - Use rolling window if you think relationships change over time
                - Use expanding window if relationships are stable
            gap_months: Months of data to skip between train and test
                - Prevents leakage from overlapping data
                - Important when features use forward-looking information
            embargo_pct: Percentage of test set to remove from beginning
                - Used when target variable overlaps (e.g., 3-year returns)
                - 0.0 = no embargo, 0.1 = remove first 10% of test period
        """
        self.n_splits = n_splits
        self.min_train_months = min_train_years * 12
        self.max_train_months = max_train_years * 12 if max_train_years else None
        self.gap_months = gap_months
        self.embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ):
        """
        Generate chronologically-ordered train/test splits.

        How it works:
        -------------
        1. Start with minimum training period (e.g., 15 years of data)
        2. First test set comes immediately after
        3. For each subsequent split:
           - Expanding: Training grows, test moves forward
           - Rolling: Training window slides forward at fixed size

        Visual example (expanding window, 3 splits):

        Time ─────────────────────────────────────►
             |----Train 1----|--Test 1--|
             |------Train 2------|--Test 2--|
             |--------Train 3--------|--Test 3--|

        Args:
            X: Feature matrix, sorted by time (oldest first)
            y: Target variable (not used internally, kept for sklearn compatibility)

        Yields:
            Tuple of (train_indices, test_indices) for each split

        Raises:
            ValueError: If dataset is too small for the requested splits
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate how big each test fold should be
        # We need enough room for training + all test folds
        available_for_testing = n_samples - self.min_train_months - self.gap_months
        fold_size = available_for_testing // self.n_splits

        # Sanity check: each fold should have at least 1 year of data
        # Otherwise our metrics will be meaningless
        if fold_size < 12:
            raise ValueError(
                f"Not enough data for {self.n_splits} cross-validation splits. "
                f"Currently have {n_samples} months, but need at least "
                f"{self.min_train_months + self.n_splits * 12} months "
                f"({self.min_train_months // 12} years min training + "
                f"{self.n_splits} years for testing). "
                f"\n\nSolutions:\n"
                f"  1. Reduce n_splits (try 3 instead of 5)\n"
                f"  2. Reduce min_train_years (try 10 instead of 15)\n"
                f"  3. Collect more historical data"
            )

        # Generate each split
        for split_number in range(self.n_splits):
            # Test period starts after training + any gap
            test_start_index = self.min_train_months + split_number * fold_size
            test_end_index = min(test_start_index + fold_size, n_samples)

            # Apply embargo: remove first portion of test set
            # This prevents leakage when targets overlap in time
            embargo_size = int(fold_size * self.embargo_pct)
            test_start_index += embargo_size

            # Determine training window
            if self.max_train_months:
                # Rolling window: fixed-size training period
                # Good when market relationships change over time
                train_start_index = max(0, test_start_index - self.max_train_months)
            else:
                # Expanding window: use all available history
                # Best when relationships are stable
                train_start_index = 0

            # Training ends before test (minus any gap)
            train_end_index = test_start_index - self.gap_months

            # Skip this split if training window is empty
            if train_end_index <= train_start_index:
                continue

            yield (
                indices[train_start_index:train_end_index],
                indices[test_start_index:test_end_index]
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of cross-validation splits."""
        return self.n_splits


class BaseReturnModel(ABC):
    """
    Abstract base class that all return prediction models must inherit from.

    Why use a base class?
    ---------------------
    This ensures all models (Linear, Random Forest, XGBoost, ARMA, ARIMA)
    have the same interface:
    - train(X, y) -> metrics
    - predict(X) -> predictions
    - evaluate(y_true, y_pred) -> metrics
    - save/load for persistence

    This uniformity means:
    - You can swap models without changing your code
    - The training pipeline works with any model
    - Evaluation is consistent across models

    Design pattern: This is the "Template Method" pattern - the base class
    defines the structure, subclasses fill in the specifics.
    """

    def __init__(self, name: str, model: BaseEstimator):
        """
        Initialize the base model.

        Args:
            name: Human-readable model name (e.g., "LinearRegression", "ARMA")
            model: The underlying estimator (sklearn, statsmodels, etc.)
        """
        self.name = name
        self.model = model
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Train the model on historical data.

        This method MUST be implemented by every subclass. It should:
        1. Handle missing values appropriately
        2. Fit the underlying estimator
        3. Set is_trained = True
        4. Return training metrics

        Args:
            X: Feature matrix (n_samples × n_features)
            y: Target vector (3-year forward returns)

        Returns:
            Dictionary containing at minimum:
            - 'rmse': Root Mean Squared Error
            - 'r2': R-squared (coefficient of determination)

        Raises:
            ValueError: If data is invalid or insufficient
        """
        pass  # Subclasses must implement this

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new data.

        This method MUST be implemented by every subclass. It should:
        1. Check that the model is trained
        2. Handle missing values consistently with training
        3. Return predictions in the same scale as the target

        Args:
            X: Feature matrix with same columns as training data

        Returns:
            Array of predicted 3-year forward returns

        Raises:
            ValueError: If called before train()
        """
        pass  # Subclasses must implement this

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Why multiple metrics?
        ---------------------
        Different metrics tell you different things:
        - RMSE: Penalizes large errors heavily (good for risk assessment)
        - MAE: Average error magnitude (easy to interpret)
        - R²: Fraction of variance explained (model quality)
        - Directional accuracy: How often you get the sign right (trading value)
        - Correlation: Linear relationship strength

        Args:
            y_true: Actual observed values
            y_pred: Model's predicted values

        Returns:
            Dictionary with comprehensive metrics:
            {
                'rmse': Root Mean Squared Error (lower is better)
                'mae': Mean Absolute Error (lower is better)
                'r2': R-squared, 0-1 range (higher is better)
                'directional_accuracy': Fraction of correct sign predictions
                'correlation': Pearson correlation coefficient
            }

        Note:
            Returns NaN for all metrics if no valid data points exist.
            This allows the pipeline to continue even with bad data.
        """
        # Create mask for valid (non-NaN) values in both arrays
        # We need both true and predicted to be valid
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Handle edge case: no valid data points
        if len(y_true_valid) == 0:
            self.logger.warning(
                "No valid data points for evaluation - returning NaN metrics"
            )
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'directional_accuracy': np.nan,
                'correlation': np.nan
            }

        # RMSE: Root Mean Squared Error
        # Heavily penalizes large mistakes (useful for risk management)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))

        # MAE: Mean Absolute Error
        # More interpretable: "on average, we're off by X%"
        mae = mean_absolute_error(y_true_valid, y_pred_valid)

        # R²: Coefficient of Determination
        # 1.0 = perfect, 0.0 = no better than predicting mean
        # Can be negative if model is worse than horizontal line
        r2 = r2_score(y_true_valid, y_pred_valid)

        # Directional Accuracy: How often do we predict the right direction?
        # This is what matters for trading decisions (buy vs sell)
        actual_direction = np.sign(y_true_valid)
        predicted_direction = np.sign(y_pred_valid)
        directional_accuracy = np.mean(
            actual_direction == predicted_direction
        )

        # Correlation: Strength of linear relationship
        # Different from R² - correlation of 0.5 means R² = 0.25
        correlation = np.corrcoef(y_true_valid, y_pred_valid)[0, 1]

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation
        }

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Extract feature importance from the model if available.

        How it works:
        - Tree models (RF, XGBoost): Have built-in feature_importances_
        - Linear models: Use absolute coefficient values
        - Other models: Return None if no importance available

        Returns:
            Series mapping feature names to importance scores (higher = more important)
            or None if the model doesn't provide importance information

        Example:
            >>> importance = model.get_feature_importance()
            >>> print(importance.head(5))  # Top 5 features
        """
        # Tree-based models have explicit feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_

        # Linear models: coefficient magnitude indicates importance
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)

        # Model doesn't provide importance information
        else:
            return None

        # Sort by importance (highest first)
        if self.feature_names:
            return pd.Series(
                importances, index=self.feature_names
            ).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)

    def save(self, filepath: str):
        """
        Save the trained model to disk using pickle.

        What gets saved:
        - The trained model/estimator
        - Feature names (for interpretation later)
        - Target name (for reference)
        - Training status flag

        Args:
            filepath: Path where to save the model (.pkl extension recommended)

        Note:
            Subclasses with additional parameters (like ARMA order)
            should override this method to save those too.
        """
        import pickle

        with open(filepath, 'wb') as file:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'is_trained': self.is_trained
            }, file)
        self.logger.info(f"Saved model to {filepath}")

    def load(self, filepath: str):
        """
        Load a previously saved model from disk.

        Args:
            filepath: Path to the saved model file

        Note:
            Subclasses with additional parameters should override
            to restore their specific attributes.
        """
        import pickle

        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.target_name = data['target_name']
            self.is_trained = data['is_trained']
        self.logger.info(f"Loaded model from {filepath}")


class ModelEnsemble:
    """
    Combines multiple models to make better predictions than any single model.

    Why use an ensemble?
    --------------------
    Different models capture different patterns:
    - Linear: Catches simple, interpretable relationships
    - Random Forest: Captures non-linear interactions
    - XGBoost: Often has the best raw performance
    - ARMA: Models time series momentum and mean reversion

    By averaging their predictions, we:
    1. Reduce variance (less sensitive to any one model's quirks)
    2. Reduce bias (cover more types of patterns)
    3. Improve robustness (one model failing doesn't break everything)

    This is the "wisdom of crowds" applied to machine learning.

    Weighting strategies:
    ---------------------
    - Equal weighting: Simple, often hard to beat
    - Performance-based: Weight by past R² or inverse RMSE
    - Domain knowledge: Give more weight to models you trust
    """

    def __init__(
        self,
        models: Dict[str, BaseReturnModel],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ensemble with multiple models.

        Args:
            models: Dictionary mapping model names to trained model instances
                Example: {"linear": LinearReturnModel(), "rf": RandomForestReturnModel()}
            weights: Optional importance for each model
                - If None: All models weighted equally
                - If provided: Should sum to 1.0 (will be normalized if not)
                - Higher weight = more influence on final prediction

        Example:
            >>> ensemble = ModelEnsemble({
            ...     "linear": linear_model,
            ...     "rf": rf_model,
            ...     "xgb": xgb_model
            ... })
            >>> predictions = ensemble.predict(X_test)
        """
        self.models = models
        # Default to equal weighting if not specified
        self.weights = weights or {name: 1.0 for name in models}
        self.logger = logging.getLogger(__name__)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions by averaging individual model predictions.

        How it works:
        1. Ask each trained model for its prediction
        2. Weight each prediction by the model's assigned weight
        3. Compute weighted average as final prediction

        Args:
            X: Feature matrix for prediction

        Returns:
            Array of ensemble predictions (weighted average)

        Raises:
            ValueError: If no models in the ensemble have been trained

        Note:
            Only trained models (is_trained=True) contribute to the prediction.
            This allows you to include models in the ensemble that haven't been
            trained yet without breaking things.
        """
        individual_predictions = []
        model_weights = []

        # Collect predictions from all trained models
        for model_name, model in self.models.items():
            if model.is_trained:
                prediction = model.predict(X)
                individual_predictions.append(prediction)
                model_weights.append(self.weights.get(model_name, 1.0))

        # Sanity check: need at least one trained model
        if not individual_predictions:
            raise ValueError(
                "Cannot make ensemble prediction: no trained models available.\n"
                f"Models in ensemble: {list(self.models.keys())}\n"
                "Make sure to call train() on at least one model first."
            )

        # Stack predictions and weights for vectorized averaging
        individual_predictions = np.array(individual_predictions)
        model_weights = np.array(model_weights)

        # Normalize weights to sum to 1.0
        model_weights = model_weights / model_weights.sum()

        # Compute weighted average across models
        ensemble_prediction = np.average(
            individual_predictions, axis=0, weights=model_weights
        )
        return ensemble_prediction

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all individual models and the ensemble.

        This is useful for:
        - Understanding which models contribute most
        - Checking if the ensemble beats individual models
        - Diagnosing if one model is dragging down performance

        Args:
            X: Feature matrix for evaluation
            y: True target values for computing metrics

        Returns:
            Nested dictionary with metrics for each model:
            {
                "linear": {"rmse": ..., "r2": ..., ...},
                "rf": {"rmse": ..., "r2": ..., ...},
                "ensemble": {"rmse": ..., "r2": ..., ...},
                ...
            }

        Tip:
            Compare ensemble metrics to individual models.
            If ensemble is worse, check if one model is particularly bad
            and consider removing it or reducing its weight.
        """
        results = {}

        # Evaluate each individual model
        for model_name, model in self.models.items():
            if model.is_trained:
                predictions = model.predict(X)
                results[model_name] = model.evaluate(y.values, predictions)
            else:
                self.logger.debug(f"Skipping untrained model: {model_name}")

        # Evaluate the ensemble prediction
        ensemble_predictions = self.predict(X)
        # Use first available model's evaluate method (they all use same metrics)
        reference_model = list(self.models.values())[0]
        results['ensemble'] = reference_model.evaluate(y.values, ensemble_predictions)

        return results
