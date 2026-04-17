"""
Model Training and Evaluation Pipeline.

This module orchestrates the lifecycle of machine learning models:
1. Loading engineered features.
2. Setting up time series cross-validation (walk-forward).
3. Training multiple model types (Ensemble, Linear, RF, XGBoost).
4. Aggregating performance metrics.
5. Generating future predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import logging
import json
import pickle

from config.settings import DATA_FEATURES_PATH, MODEL_CONFIG, MARKETS
from .base import TimeSeriesCrossValidator
from .implementations import create_model

# Set up module logger
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Main orchestrator for model training, validation, and prediction.
    """

    def __init__(
        self,
        model_selection: str = "ensemble",
        target_markets: Optional[List[str]] = None,
        feature_data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the training pipeline.
        """
        self.model_selection = model_selection
        self.markets = self._initialize_markets(target_markets)
        self.feature_dir = Path(feature_data_dir or DATA_FEATURES_PATH)

        # In-memory storage for trained models and results
        self.trained_models: Dict[str, Dict[str, Any]] = {}
        self.cross_val_metrics: Dict[str, Any] = {}
        self.feature_matrix: Optional[pd.DataFrame] = None

        self.logger = logging.getLogger(__name__)

    def _initialize_markets(self, markets: Optional[List[str]]) -> List[str]:
        """Validate and return the list of target markets."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets

    def load_feature_matrix(self, filename: str = "features.csv") -> bool:
        """
        Load the engineered feature matrix from disk.
        """
        file_path = self.feature_dir / filename

        if not file_path.exists():
            self.logger.error(f"Feature matrix not found at {file_path}.")
            return False

        self.feature_matrix = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded feature matrix: {self.feature_matrix.shape}")
        return True

    def execute_training_cycle(self) -> Dict[str, Any]:
        """
        Runs the full training and cross-validation cycle for all markets.
        """
        self.logger.info(f"Starting training cycle for model type: {self.model_selection}")

        if self.feature_matrix is None:
            if not self.load_feature_matrix():
                return {}

        # Configure the cross-validator
        cross_validator = TimeSeriesCrossValidator(
            n_splits=MODEL_CONFIG["n_splits"],
            min_train_years=MODEL_CONFIG["min_train_years"],
            max_train_years=MODEL_CONFIG.get("max_train_years")
        )

        pipeline_results = {}

        for market in self.markets:
            self.logger.info(f"\n{'=' * 60}\nTRAINING MARKET: {market}\n{'=' * 60}")
            market_summary = self._train_single_market(market, cross_validator)
            pipeline_results[market] = market_summary

        self.logger.info("\n" + "=" * 60 + "\nTraining cycle completed.\n" + "=" * 60)
        return pipeline_results

    def _train_single_market(
        self, 
        market: str, 
        cv: TimeSeriesCrossValidator
    ) -> Dict[str, Any]:
        """
        Handles the training workflow for a specific market.
        """
        target_col = f"{market}_target_return"

        if target_col not in self.feature_matrix.columns:
            self.logger.error(f"Target column '{target_col}' not found.")
            return {}

        # 1. Feature Selection: Pick relevant columns
        X_market = self._select_features_for_market(market)
        y = self.feature_matrix[target_col]
        
        self.logger.info(f"Using {len(X_market.columns)} features for {market}")

        # 2. Data Cleaning: Align X and y and drop NaNs
        combined = X_market.join(y).dropna()
        
        if len(combined) < (cv.min_train_months + 12):
            self.logger.error(f"Insufficient data for {market} after NaN removal ({len(combined)} rows)")
            return {}

        X_trainable = combined.drop(columns=[target_col])
        y_trainable = combined[target_col]

        # 3. Dynamic CV adjustment
        max_possible_splits = (len(X_trainable) - cv.min_train_months) // 12
        actual_n_splits = 0
        fold_metrics_history = []
        
        if max_possible_splits >= 2:
            actual_n_splits = min(cv.n_splits, max_possible_splits)
            market_cv = TimeSeriesCrossValidator(
                n_splits=actual_n_splits,
                min_train_years=cv.min_train_months // 12
            )
            
            for fold_num, (train_idx, test_idx) in enumerate(market_cv.split(X_trainable)):
                X_train, X_test = X_trainable.iloc[train_idx], X_trainable.iloc[test_idx]
                y_train, y_test = y_trainable.iloc[train_idx], y_trainable.iloc[test_idx]
                fold_results = self._evaluate_on_fold(X_train, y_train, X_test, y_test)
                fold_metrics_history.append(fold_results)

        # 4. Final Training
        self.logger.info(f"Training final '{self.model_selection}' models on {len(X_trainable)} samples...")
        final_models = self._instantiate_models()
        self.trained_models[market] = {}
        
        for name, model in final_models.items():
            try:
                model.train(X_trainable, y_trainable)
                self.trained_models[market][name] = model
            except Exception as e:
                self.logger.error(f"  ✗ {name} training failed: {str(e)}")

        aggregated_metrics = self._aggregate_metrics(fold_metrics_history) if fold_metrics_history else {}
        self.cross_val_metrics[market] = aggregated_metrics

        return {
            "cv_metrics": aggregated_metrics,
            "n_samples": len(X_trainable),
            "model_count": len(self.trained_models[market]),
            "actual_splits": actual_n_splits
        }

    def _select_features_for_market(self, market: str) -> pd.DataFrame:
        """
        Strict feature selection to prevent data leakage and ensure consistency.
        """
        # All columns except target columns
        X = self.feature_matrix.drop(
            columns=[col for col in self.feature_matrix.columns if "_target_return" in col]
        )
        
        other_markets = [m for m in self.markets if m != market]
        
        # We want to exclude:
        # 1. Local features of OTHER markets (e.g. Europe_volatility when training USA)
        # 2. Macro features of countries that don't belong to this market
        #    (Unless we want global macro, but let's be strict for now to keep it clean)
        
        cols_to_exclude = []
        for col in X.columns:
            # Exclude other markets' local features
            for other in other_markets:
                if col.startswith(f"{other}_"):
                    cols_to_exclude.append(col)
                    break
        
        return X.drop(columns=cols_to_exclude)

    def _instantiate_models(self) -> Dict[str, Any]:
        models = {}
        types = ["linear", "rf", "xgb", "arma", "arima"] if self.model_selection in ["ensemble", "all"] else [self.model_selection]
        for m_type in types:
            try: models[m_type] = create_model(m_type)
            except ValueError as e: self.logger.error(str(e))
        return models

    def _evaluate_on_fold(self, X_train, y_train, X_test, y_test) -> Dict[str, Dict]:
        models = self._instantiate_models()
        results = {}
        for name, model in models.items():
            try:
                model.train(X_train, y_train)
                preds = model.predict(X_test)
                results[name] = model.evaluate(y_test.values, preds)
            except Exception:
                results[name] = {"rmse": np.nan, "r2": np.nan}
        return results

    def _aggregate_metrics(self, history: List[Dict]) -> Dict[str, Dict]:
        if not history: return {}
        model_names = history[0].keys()
        summary = {}
        for name in model_names:
            model_folds = [h[name] for h in history]
            metric_names = model_folds[0].keys()
            summary[name] = {m: np.nanmean([f[m] for f in model_folds]) for m in metric_names}
        return summary

    def generate_forecasts(self) -> Dict[str, pd.DataFrame]:
        """
        Uses trained models to predict future returns.
        Guarantees feature consistency by using the models' stored feature names.
        """
        self.logger.info("Generating forecasts...")
        if self.feature_matrix is None and not self.load_feature_matrix(): return {}
        if not self.trained_models and not self.load_trained_models(): return {}

        forecasts = {}
        # Prediction row (most recent data point)
        latest_row = self.feature_matrix.iloc[-1:]

        for market in self.markets:
            if market not in self.trained_models: continue
            
            predictions = []
            for name, model in self.trained_models[market].items():
                try:
                    # Use stored feature names to filter the input data!
                    # This is the key fix for the "unseen feature names" error.
                    if model.feature_names:
                        X_pred = latest_row[model.feature_names]
                    else:
                        # Fallback for models that don't store names (shouldn't happen)
                        X_pred = self._select_features_for_market(market).iloc[-1:]
                        
                    val = model.predict(X_pred)[0]
                    predictions.append({"model": name, "predicted_return": val})
                except Exception as e:
                    self.logger.error(f"Forecast failed for {market}/{name}: {str(e)}")
            
            if predictions:
                df = pd.DataFrame(predictions)
                df["market"] = market
                forecasts[market] = df
        return forecasts

    def save_trained_models(self, directory: Optional[Union[str, Path]] = None):
        save_path = Path(directory or self.feature_dir.parent / "models")
        save_path.mkdir(parents=True, exist_ok=True)
        for market, models in self.trained_models.items():
            market_dir = save_path / market
            market_dir.mkdir(exist_ok=True)
            for name, model in models.items():
                model.save(str(market_dir / f"{name}.pkl"))
        metadata = {"markets": self.markets, "model_selection": self.model_selection, "timestamp": datetime.now().isoformat()}
        with open(save_path / "metadata.json", "w") as f: json.dump(metadata, f, indent=2)

    def load_trained_models(self, directory: Optional[Union[str, Path]] = None) -> bool:
        load_path = Path(directory or self.feature_dir.parent / "models")
        metadata_file = load_path / "metadata.json"
        if not metadata_file.exists(): return False
        with open(metadata_file, "r") as f: meta = json.load(f)
        for market in meta.get("markets", self.markets):
            market_dir = load_path / market
            if not market_dir.exists(): continue
            self.trained_models[market] = {}
            for pkl_file in market_dir.glob("*.pkl"):
                name = pkl_file.stem
                try:
                    from .implementations import LinearReturnModel, RandomForestReturnModel, XGBoostReturnModel, ARMAReturnModel, ARIMAReturnModel
                    class_map = {"linear": LinearReturnModel, "rf": RandomForestReturnModel, "xgb": XGBoostReturnModel, "arma": ARMAReturnModel, "arima": ARIMAReturnModel}
                    if name in class_map:
                        model_inst = class_map[name]()
                        model_inst.load(str(pkl_file))
                        self.trained_models[market][name] = model_inst
                except Exception as e: self.logger.error(f"Failed to load {market}/{name}: {str(e)}")
        return len(self.trained_models) > 0

    def print_evaluation_summary(self, results: Dict[str, Any]):
        print("\n" + "=" * 60 + "\nMODEL PERFORMANCE SUMMARY\n" + "=" * 60)
        for market, data in results.items():
            if not data: continue
            print(f"\nMARKET: {market} ({data.get('actual_splits', 0)} CV Splits)")
            metrics = data.get("cv_metrics", {})
            if not metrics:
                print("  [!] No CV performed (limited data).")
                continue
            for name, m in metrics.items():
                acc = m.get('directional_accuracy', np.nan)
                acc_str = f"{acc:.1%}" if not np.isnan(acc) else "N/A"
                print(f"  {name:10s} | R²: {m.get('r2', np.nan):6.3f} | RMSE: {m.get('rmse', np.nan):6.2f} | Acc: {acc_str}")
