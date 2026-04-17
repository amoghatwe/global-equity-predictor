"""
Feature engineering pipeline for creating ML features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
from pathlib import Path

from config.settings import (
    DATA_PROCESSED_PATH,
    DATA_FEATURES_PATH,
    MARKETS,
    FORECAST_HORIZON_MONTHS,
    FEATURE_CATEGORIES
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Pipeline for engineering features from raw data.
    """
    
    def __init__(
        self,
        markets: Optional[List[str]] = None,
        processed_data_path: Optional[str] = None
    ):
        self.markets = self._parse_markets(markets)
        self.processed_data_path = processed_data_path or DATA_PROCESSED_PATH
        self.features_path = DATA_FEATURES_PATH
        
        self.processed_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.targets: Optional[pd.DataFrame] = None
        
        self.logger = logging.getLogger(__name__)
        
    def _parse_markets(self, markets: Optional[List[str]]) -> List[str]:
        """Parse market specification."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets
    
    def load_processed_data(self, filename: str = "processed_data.csv"):
        """Load processed data from file."""
        file_path = Path(self.processed_data_path) / filename
        
        if not file_path.exists():
            self.logger.error(f"Processed data not found: {file_path}")
            return False
        
        self.processed_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded processed data: {self.processed_data.shape}")
        return True
    
    def create_features(self) -> pd.DataFrame:
        """
        Create all features from processed data.
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        self.logger.info("Creating features...")
        
        if self.processed_data is None:
            if not self.load_processed_data():
                return pd.DataFrame()
        
        features_dict = {}
        
        # 1. Valuation features
        self.logger.info("Creating valuation features...")
        valuation = self._create_valuation_features()
        if not valuation.empty:
            features_dict['valuation'] = valuation
        
        # 2. Growth features
        self.logger.info("Creating growth features...")
        growth = self._create_growth_features()
        if not growth.empty:
            features_dict['growth'] = growth
        
        # 3. Inflation features
        self.logger.info("Creating inflation features...")
        inflation = self._create_inflation_features()
        if not inflation.empty:
            features_dict['inflation'] = inflation
        
        # 4. Interest rate features
        self.logger.info("Creating interest rate features...")
        rates = self._create_rate_features()
        if not rates.empty:
            features_dict['rates'] = rates
        
        # 5. Credit features
        self.logger.info("Creating credit features...")
        credit = self._create_credit_features()
        if not credit.empty:
            features_dict['credit'] = credit
        
        # 6. Momentum features
        self.logger.info("Creating momentum features...")
        momentum = self._create_momentum_features()
        if not momentum.empty:
            features_dict['momentum'] = momentum
        
        # Combine all features
        if not features_dict:
            self.logger.error("No features created")
            return pd.DataFrame()
        
        # Start with first feature set
        combined = list(features_dict.values())[0]
        
        # Join remaining
        for name, df in list(features_dict.items())[1:]:
            combined = combined.join(df, how='outer')
        
        # Clean up
        combined = combined.dropna(how='all')
        combined = combined.sort_index()
        
        self.features = combined
        self.logger.info(f"Created {len(combined.columns)} features across {len(combined)} periods")
        
        return combined
    
    def _create_valuation_features(self) -> pd.DataFrame:
        """Create valuation-based features."""
        features = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        for market in self.markets:
            # Try to find price data
            price_col = f"equity_{market}"
            
            if price_col in data.columns:
                prices = data[price_col]
                
                # Price-to-moving average (momentum/valuation hybrid)
                for window in [12, 36, 60]:
                    ma = prices.rolling(window=window).mean()
                    features[f"{market}_price_ma{window}_ratio"] = prices / ma
                
                # Long-term trend deviation (avoid division by zero)
                long_ma = prices.rolling(window=60).mean()
                with np.errstate(divide='ignore', invalid='ignore'):
                    features[f"{market}_price_deviation_60m"] = np.where(
                        long_ma != 0,
                        (prices - long_ma) / long_ma * 100,
                        np.nan
                    )
        
        return features
    
    def _create_growth_features(self) -> pd.DataFrame:
        """Create economic growth features."""
        features = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        # GDP growth trends
        for col in data.columns:
            if 'gdp_growth' in col:
                # Trend (3-year moving average)
                features[f"{col}_trend"] = data[col].rolling(36).mean()
                
                # Deviation from trend
                trend = data[col].rolling(36).mean()
                features[f"{col}_deviation"] = data[col] - trend
                
                # Year-over-year change
                features[f"{col}_yoy_change"] = data[col].diff(12)
        
        # Industrial production
        for col in data.columns:
            if 'industrial_production' in col:
                features[f"{col}_growth"] = data[col].pct_change(12) * 100
                features[f"{col}_trend"] = data[col].rolling(36).mean()
        
        return features
    
    def _create_inflation_features(self) -> pd.DataFrame:
        """Create inflation-related features."""
        features = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        # CPI inflation
        for col in data.columns:
            if 'cpi_yoy' in col or 'inflation' in col:
                # Inflation level
                features[f"{col}_level"] = data[col]
                
                # Inflation trend (3-year)
                features[f"{col}_trend"] = data[col].rolling(36).mean()
                
                # Inflation deviation from trend
                trend = data[col].rolling(36).mean()
                features[f"{col}_deviation"] = data[col] - trend
        
        # Real interest rates
        if 'us_data_yield_10y' in data.columns:
            for col in data.columns:
                if 'inflation' in col.lower() and col in data.columns:
                    inflation_data = data[col]
                    if inflation_data is not None and not inflation_data.empty:
                        features['real_rate_10y'] = data['us_data_yield_10y'] - inflation_data
                        break
        
        return features
    
    def _create_rate_features(self) -> pd.DataFrame:
        """Create interest rate features."""
        features = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        # US rates (most important globally)
        if 'us_data_yield_10y' in data.columns:
            features['us_yield_10y_level'] = data['us_data_yield_10y']
            features['us_yield_10y_trend'] = data['us_data_yield_10y'].rolling(36).mean()
            features['us_yield_10y_change_12m'] = data['us_data_yield_10y'].diff(12)
        
        if 'us_data_fed_funds' in data.columns:
            features['fed_funds_level'] = data['us_data_fed_funds']
            features['fed_funds_change_12m'] = data['us_data_fed_funds'].diff(12)
        
        # Yield curve
        if 'us_data_yield_curve_spread' in data.columns:
            features['yield_curve_spread'] = data['us_data_yield_curve_spread']
            features['yield_curve_spread_change_12m'] = data['us_data_yield_curve_spread'].diff(12)
        elif 'us_data_yield_10y' in data.columns and 'us_data_yield_2y' in data.columns:
            spread = data['us_data_yield_10y'] - data['us_data_yield_2y']
            features['yield_curve_spread'] = spread
            features['yield_curve_spread_change_12m'] = spread.diff(12)
        
        return features
    
    def _create_credit_features(self) -> pd.DataFrame:
        """Create credit/money supply features."""
        features = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        # M2 growth
        for col in data.columns:
            if 'm2' in col.lower():
                features[f"{col}_growth_12m"] = data[col].pct_change(12) * 100
                features[f"{col}_growth_trend"] = data[col].pct_change(12).rolling(36).mean() * 100
        
        # Credit to GDP
        for col in data.columns:
            if 'credit' in col.lower():
                features[f"{col}_level"] = data[col]
                features[f"{col}_change_12m"] = data[col].diff(12)
        
        return features
    
    def _create_momentum_features(self) -> pd.DataFrame:
        """Create momentum/volatility features."""
        features = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        for market in self.markets:
            price_col = f"equity_{market}"
            
            if price_col in data.columns:
                prices = data[price_col]
                
                # Trailing returns
                for months in [1, 3, 6, 12]:
                    returns = prices.pct_change(months) * 100
                    features[f"{market}_return_{months}m"] = returns
                
                # Volatility
                monthly_returns = prices.pct_change()
                for window in [6, 12]:
                    vol = monthly_returns.rolling(window).std() * np.sqrt(12) * 100
                    features[f"{market}_volatility_{window}m"] = vol
        
        # VIX (market fear)
        if 'us_data_vix' in data.columns:
            features['vix_level'] = data['us_data_vix']
            features['vix_trend'] = data['us_data_vix'].rolling(12).mean()
        
        return features
    
    def create_targets(self) -> pd.DataFrame:
        """
        Create target variables (forward returns).
        
        Returns:
            pd.DataFrame: Target matrix
        """
        self.logger.info("Creating target variables...")
        
        if self.processed_data is None:
            if not self.load_processed_data():
                return pd.DataFrame()
        
        targets = pd.DataFrame(index=self.processed_data.index)
        data = self.processed_data
        
        for market in self.markets:
            price_col = f"equity_{market}"
            
            if price_col in data.columns:
                prices = data[price_col]
                
                # Calculate forward returns (target)
                future_price = prices.shift(-FORECAST_HORIZON_MONTHS)
                total_return = (future_price / prices) - 1
                
                # Annualize
                annualized_return = ((1 + total_return) ** (12 / FORECAST_HORIZON_MONTHS)) - 1
                
                targets[f"{market}_return_3y"] = annualized_return * 100
                
                self.logger.info(f"Created target for {market}: "
                                f"{targets[f'{market}_return_3y'].notna().sum()} valid observations")
        
        self.targets = targets
        return targets
    
    def merge_features_targets(self, features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        """
        Merge features and targets into training dataset.
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            
        Returns:
            pd.DataFrame: Combined dataset with targets
        """
        self.logger.info("Merging features and targets...")
        
        # Align dates
        combined = features.join(targets, how='inner')
        
        # Drop rows where any target is missing
        target_cols = [col for col in combined.columns if '_return_3y' in col]
        combined = combined.dropna(subset=target_cols, how='any')
        
        self.logger.info(f"Final dataset: {len(combined)} rows, {len(combined.columns)} columns")
        
        return combined
    
    def save(self, data: pd.DataFrame, filename: str = "features.csv"):
        """Save features data."""
        output_path = self.features_path / filename
        data.to_csv(output_path, index=True)
        self.logger.info(f"Saved features to {output_path}")
    
    def load(self, filename: str = "features.csv") -> pd.DataFrame:
        """Load features data."""
        input_path = self.features_path / filename
        
        if not input_path.exists():
            self.logger.error(f"Features not found: {input_path}")
            return pd.DataFrame()
        
        data = pd.read_csv(input_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded features: {data.shape}")
        return data
