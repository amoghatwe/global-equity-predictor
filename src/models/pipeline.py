"""
Model training pipeline with walk-forward validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging
import json

from config.settings import (
    DATA_FEATURES_PATH,
    MODEL_CONFIG,
    MARKETS
)
from .base import TimeSeriesCrossValidator, ModelEnsemble
from .implementations import create_model

logger = logging.getLogger(__name__)


class ModelPipeline:
    """
    Pipeline for training and evaluating models with time series validation.
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        markets: Optional[List[str]] = None,
        features_path: Optional[str] = None
    ):
        """
        Initialize model pipeline.
        
        Args:
            model_type: 'linear', 'rf', 'xgb', or 'ensemble'
            markets: List of markets to train on
            features_path: Path to features data
        """
        self.model_type = model_type
        self.markets = self._parse_markets(markets)
        self.features_path = Path(features_path or DATA_FEATURES_PATH)
        
        self.models: Dict[str, Dict[str, Any]] = {}  # market -> {model_type: model}
        self.cv_results: Dict[str, List[Dict]] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        
        self.data: Optional[pd.DataFrame] = None
        
        self.logger = logging.getLogger(__name__)
        
    def _parse_markets(self, markets: Optional[List[str]]) -> List[str]:
        """Parse market specification."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets
    
    def load_data(self, filename: str = "features.csv") -> bool:
        """
        Load feature data.
        
        Args:
            filename: Name of features file
            
        Returns:
            True if successful
        """
        filepath = self.features_path / filename
        
        if not filepath.exists():
            self.logger.error(f"Features file not found: {filepath}")
            return False
        
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded features: {self.data.shape}")
        return True
    
    def train(self) -> Dict[str, Any]:
        """
        Train models for all markets using walk-forward validation.
        
        Returns:
            Dictionary with training results
        """
        self.logger.info("Starting model training pipeline...")
        
        if self.data is None:
            if not self.load_data():
                return {}
        
        # Setup cross-validator
        cv = TimeSeriesCrossValidator(
            n_splits=MODEL_CONFIG['n_splits'],
            min_train_years=MODEL_CONFIG['min_train_years'],
            max_train_years=MODEL_CONFIG.get('max_train_years'),
            gap_months=0,
            embargo_pct=0.0
        )
        
        results = {}
        
        for market in self.markets:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training models for {market}")
            self.logger.info(f"{'='*60}")
            
            market_results = self._train_market(market, cv)
            results[market] = market_results
        
        self.logger.info("\nTraining completed for all markets")
        return results
    
    def _train_market(self, market: str, cv: TimeSeriesCrossValidator) -> Dict[str, Any]:
        """
        Train models for a single market.
        
        Args:
            market: Market name
            cv: Cross-validator
            
        Returns:
            Training results for market
        """
        # Get target column
        target_col = f"{market}_return_3y"
        
        if target_col not in self.data.columns:
            self.logger.error(f"Target column not found: {target_col}")
            return {}
        
        # Prepare data
        y = self.data[target_col]
        X = self.data.drop(columns=[col for col in self.data.columns if '_return_3y' in col])
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < cv.min_train_months + 24:
            self.logger.error(f"Insufficient data for {market}: {len(X)} samples")
            return {}
        
        self.logger.info(f"Data: {len(X)} samples, {X.shape[1]} features")
        
        # Store models for this market
        self.models[market] = {}
        self.cv_results[market] = []
        
        # Cross-validation
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            self.logger.info(f"\nFold {fold_idx + 1}/{cv.n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            self.logger.info(f"  Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
            self.logger.info(f"  Test:  {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")
            
            # Train models
            fold_result = self._train_fold(X_train, y_train, X_test, y_test, fold_idx)
            fold_metrics.append(fold_result)
        
        # Train final model on all data
        self.logger.info(f"\nTraining final model on all {len(X)} samples...")
        final_models = self._create_models()
        
        for model_name, model in final_models.items():
            try:
                metrics = model.train(X, y)
                self.models[market][model_name] = model
                self.logger.info(f"  {model_name}: R² = {metrics['r2']:.3f}")
            except Exception as e:
                self.logger.error(f"  {model_name}: Training failed - {str(e)}")
        
        # Aggregate CV results
        avg_metrics = self._aggregate_cv_metrics(fold_metrics)
        
        self.logger.info(f"\nCross-validation results for {market}:")
        for model_name, metrics in avg_metrics.items():
            self.logger.info(f"  {model_name}: RMSE={metrics['rmse']:.2f}, "
                           f"R²={metrics['r2']:.3f}, "
                           f"Dir.Acc={metrics['directional_accuracy']:.1%}")
        
        return {
            'cv_metrics': avg_metrics,
            'final_models': list(self.models[market].keys()),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
    
    def _create_models(self) -> Dict[str, Any]:
        """Create model instances based on configuration."""
        models = {}
        
        if self.model_type == 'ensemble' or self.model_type == 'all':
            models['linear'] = create_model('linear')
            models['rf'] = create_model('rf')
            models['xgb'] = create_model('xgb')
        elif self.model_type in ['linear', 'rf', 'xgb']:
            models[self.model_type] = create_model(self.model_type)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return models
    
    def _train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fold_idx: int
    ) -> Dict[str, Dict]:
        """
        Train and evaluate models on a single CV fold.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            fold_idx: Fold index
            
        Returns:
            Dictionary of metrics for each model
        """
        fold_metrics = {}
        models = self._create_models()
        
        for model_name, model in models.items():
            try:
                # Train
                model.train(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                metrics = model.evaluate(y_test.values, y_pred)
                fold_metrics[model_name] = metrics
                
            except Exception as e:
                self.logger.error(f"    {model_name}: Error - {str(e)}")
                fold_metrics[model_name] = {
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'directional_accuracy': np.nan
                }
        
        return fold_metrics
    
    def _aggregate_cv_metrics(self, fold_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across CV folds."""
        if not fold_metrics:
            return {}
        
        # Get model names from first fold
        model_names = list(fold_metrics[0].keys())
        
        aggregated = {}
        
        for model_name in model_names:
            # Collect metrics across folds
            metrics_by_fold = [fold[model_name] for fold in fold_metrics]
            
            # Calculate mean and std for each metric
            metric_names = list(metrics_by_fold[0].keys())
            aggregated[model_name] = {}
            
            for metric_name in metric_names:
                values = [m[metric_name] for m in metrics_by_fold]
                aggregated[model_name][metric_name] = np.nanmean(values)
                aggregated[model_name][f"{metric_name}_std"] = np.nanstd(values)
        
        return aggregated
    
    def predict(self) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions for current date.
        
        Returns:
            Dictionary of predictions by market
        """
        self.logger.info("Generating predictions...")
        
        if self.data is None:
            if not self.load_data():
                return {}
        
        if not self.models:
            if not self.load_models():
                self.logger.error("No models available for prediction")
                return {}
        
        predictions = {}
        
        # Get latest feature values
        X_latest = self.data.iloc[-1:].copy()
        
        # Remove target columns
        X_latest = X_latest.drop(columns=[col for col in X_latest.columns if '_return_3y' in col])
        
        for market in self.markets:
            if market not in self.models:
                self.logger.warning(f"No models trained for {market}")
                continue
            
            market_predictions = {}
            
            # Get predictions from each model
            for model_name, model in self.models[market].items():
                try:
                    pred = model.predict(X_latest)[0]
                    market_predictions[model_name] = pred
                except Exception as e:
                    self.logger.error(f"Prediction failed for {market}/{model_name}: {str(e)}")
            
            # Calculate ensemble prediction
            if market_predictions:
                predictions[market] = pd.DataFrame({
                    'model': list(market_predictions.keys()),
                    'predicted_return': list(market_predictions.values())
                })
                predictions[market]['market'] = market
        
        return predictions
    
    def save_models(self, output_dir: Optional[str] = None):
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        if output_dir is None:
            output_dir = self.features_path.parent / "models"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving models to {output_path}...")
        
        for market, models in self.models.items():
            market_dir = output_path / market
            market_dir.mkdir(exist_ok=True)
            
            for model_name, model in models.items():
                filepath = market_dir / f"{model_name}.pkl"
                model.save(str(filepath))
        
        # Save metadata
        metadata = {
            'markets': self.markets,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'n_models': sum(len(models) for models in self.models.values())
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved {metadata['n_models']} models")
    
    def load_models(self, input_dir: Optional[str] = None) -> bool:
        """
        Load trained models from disk.
        
        Args:
            input_dir: Directory to load models from
            
        Returns:
            bool: True if models were loaded successfully
        """
        if input_dir is None:
            input_dir = self.features_path.parent / "models"
        
        input_path = Path(input_dir)
        
        if not input_path.exists():
            self.logger.error(f"Models directory not found: {input_path}")
            return False
        
        metadata_file = input_path / "metadata.json"
        if not metadata_file.exists():
            self.logger.error(f"Metadata file not found: {metadata_file}")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(f"Loading {metadata['n_models']} models from {input_path}...")
        
        for market in metadata.get('markets', self.markets):
            market_dir = input_path / market
            if not market_dir.exists():
                self.logger.warning(f"Market directory not found: {market_dir}")
                continue
            
            self.models[market] = {}
            
            for model_file in market_dir.glob("*.pkl"):
                model_name = model_file.stem
                try:
                    import pickle
                    with open(model_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    sklearn_model = data['model']
                    
                    if model_name == 'xgb':
                        from .implementations import XGBoostReturnModel
                        model_instance = XGBoostReturnModel()
                    elif model_name == 'rf':
                        from .implementations import RandomForestReturnModel
                        model_instance = RandomForestReturnModel()
                    elif model_name == 'linear':
                        from .implementations import LinearReturnModel
                        model_instance = LinearReturnModel()
                    else:
                        self.logger.error(f"Unknown model name: {model_name}")
                        continue
                    
                    model_instance.model = sklearn_model
                    model_instance.feature_names = data['feature_names']
                    model_instance.target_name = data['target_name']
                    model_instance.is_trained = data['is_trained']
                    self.models[market][model_name] = model_instance
                    self.logger.debug(f"Loaded {market}/{model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load {market}/{model_name}: {str(e)}")
        
        loaded_count = sum(len(models) for models in self.models.values())
        self.logger.info(f"Loaded {loaded_count} models")
        
        return loaded_count > 0
    
    def evaluate(self, results: Dict[str, Any]):
        """
        Print evaluation summary.
        
        Args:
            results: Training results
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODEL EVALUATION SUMMARY")
        self.logger.info("="*60)
        
        for market, result in results.items():
            if not result:
                continue
            
            self.logger.info(f"\n{market}:")
            self.logger.info(f"  Samples: {result.get('n_samples', 0)}")
            self.logger.info(f"  Features: {result.get('n_features', 0)}")
            
            cv_metrics = result.get('cv_metrics', {})
            for model_name, metrics in cv_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                rmse = metrics.get('rmse', np.nan)
                r2 = metrics.get('r2', np.nan)
                dir_acc = metrics.get('directional_accuracy', np.nan)
                
                self.logger.info(f"  {model_name:15s}: RMSE={rmse:6.2f}, "
                               f"R²={r2:6.3f}, DirAcc={dir_acc:.1%}")
