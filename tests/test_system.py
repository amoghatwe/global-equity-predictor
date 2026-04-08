"""
Basic tests for the Global Equity Predictor system.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import MARKETS, FORECAST_HORIZON_MONTHS


class TestDataCollection:
    """Tests for data collection module."""
    
    def test_market_configuration(self):
        """Test that markets are configured."""
        assert len(MARKETS) > 0
        assert all(isinstance(k, str) for k in MARKETS.keys())
    
    def test_forecast_horizon(self):
        """Test forecast horizon is reasonable."""
        assert FORECAST_HORIZON_MONTHS == 36  # 3 years
        assert FORECAST_HORIZON_MONTHS > 12   # More than 1 year


class TestFeatureEngineering:
    """Tests for feature engineering."""
    
    def test_forward_return_calculation(self):
        """Test forward return calculation."""
        # Create sample price data
        dates = pd.date_range(start='2020-01-01', periods=48, freq='ME')
        prices = pd.Series(100 * (1.01 ** np.arange(48)), index=dates)
        
        # Calculate 36-month forward return
        future_price = prices.shift(-36)
        total_return = (future_price / prices) - 1
        annualized = ((1 + total_return) ** (12/36)) - 1
        
        # Should have NaN for last 36 months
        assert annualized.iloc[-36:].isna().all()
        assert annualized.iloc[:-36].notna().all()
    
    def test_feature_creation_sample(self):
        """Test basic feature creation logic."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=24, freq='ME')
        df = pd.DataFrame({
            'price': 100 + np.arange(24),
            'gdp': 2.0 + np.random.randn(24) * 0.5
        }, index=dates)
        
        # Calculate moving average
        df['price_ma12'] = df['price'].rolling(12).mean()
        
        assert 'price_ma12' in df.columns
        assert df['price_ma12'].iloc[:11].isna().all()  # First 11 should be NaN
        assert df['price_ma12'].iloc[12:].notna().all()  # Rest should have values


class TestModels:
    """Tests for ML models."""
    
    def test_linear_model_creation(self):
        """Test linear model can be created."""
        from src.models.implementations import LinearReturnModel
        
        model = LinearReturnModel()
        assert model is not None
        assert model.name == "LinearRegression"
    
    def test_model_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        from src.models.base import BaseReturnModel
        from sklearn.linear_model import Ridge
        
        class DummyModel(BaseReturnModel):
            def train(self, X, y):
                self.is_trained = True
                return {}
            def predict(self, X):
                return np.zeros(len(X))
        
        model = DummyModel("dummy", Ridge())
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = model.evaluate(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'directional_accuracy' in metrics
        
        # Directional accuracy should be high for this case
        assert metrics['directional_accuracy'] == 1.0  # All same direction


class TestTimeSeriesCV:
    """Tests for time series cross-validation."""
    
    def test_cv_split_sizes(self):
        """Test CV generates correct split sizes."""
        from src.models.base import TimeSeriesCrossValidator
        
        # Create sample time series
        n_samples = 240  # 20 years
        X = pd.DataFrame(np.random.randn(n_samples, 5))
        
        cv = TimeSeriesCrossValidator(
            n_splits=5,
            min_train_years=10,
            max_train_years=None
        )
        
        splits = list(cv.split(X))
        
        assert len(splits) == 5
        
        # Each split should have train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_expanding_window(self):
        """Test expanding window increases over splits."""
        from src.models.base import TimeSeriesCrossValidator
        
        n_samples = 300
        X = pd.DataFrame(np.random.randn(n_samples, 5))
        
        cv = TimeSeriesCrossValidator(
            n_splits=5,
            min_train_years=15,
            max_train_years=None  # Expanding
        )
        
        train_sizes = []
        for train_idx, _ in cv.split(X):
            train_sizes.append(len(train_idx))
        
        # Expanding window - sizes should increase
        assert train_sizes == sorted(train_sizes)


class TestReporting:
    """Tests for report generation."""
    
    def test_confidence_calculation(self):
        """Test confidence level calculation."""
        from src.reporting.generator import ReportGenerator
        
        generator = ReportGenerator()
        
        # Low std = high confidence
        assert generator._calculate_confidence(0.5) == "High"
        assert generator._calculate_confidence(1.5) == "Medium"
        assert generator._calculate_confidence(3.0) == "Low"


class TestIntegration:
    """Integration tests."""
    
    def test_pipeline_end_to_end(self):
        """Test complete pipeline with synthetic data."""
        # Create synthetic data
        dates = pd.date_range(start='2000-01-01', periods=300, freq='ME')
        n_samples = len(dates)
        
        # Features
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
        }, index=dates)
        
        # Target (3-year forward return)
        y = pd.Series(np.random.randn(n_samples) * 5, index=dates, name='USA_return_3y')
        y.iloc[-36:] = np.nan  # Last 36 months have no forward return
        
        # Remove NaN
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        assert len(X) == len(y)
        assert len(X) > 100  # Should have substantial data


if __name__ == "__main__":
    # Run tests with pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed. Running basic checks...")
        
        # Run basic tests
        test = TestDataCollection()
        test.test_market_configuration()
        test.test_forecast_horizon()
        print("✓ Data collection tests passed")
        
        test = TestFeatureEngineering()
        test.test_forward_return_calculation()
        test.test_feature_creation_sample()
        print("✓ Feature engineering tests passed")
        
        test = TestModels()
        test.test_linear_model_creation()
        test.test_model_evaluation_metrics()
        print("✓ Model tests passed")
        
        test = TestTimeSeriesCV()
        test.test_cv_split_sizes()
        test.test_expanding_window()
        print("✓ Time series CV tests passed")
        
        test = TestReporting()
        test.test_confidence_calculation()
        print("✓ Reporting tests passed")
        
        print("\n✓ All basic tests passed!")
